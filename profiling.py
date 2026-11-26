"""Standalone, batch-one online profiling with explicit timing boundaries.

This module loads a checkpoint and constructs every backbone feature on demand.
It does not substitute cached inference times for end-to-end timing. Run only
when other training/benchmark work is stopped; host exclusivity is not inferred.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import platform
import threading
import time

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

from .catalog import CatalogProblem
from .controllers import solve_baseline, solve_joint, solver_thread_capability
from .data import WindowDataset, file_sha256
from .experiments import read_interface, write_json
from .graphs import RollingGraph
from .memory import HistoryBank
from .protocol import ForecastLedger, MetadataSimulator, generate_manifest
from .training import load_checkpoint, resolve_device


CONTROLLERS = {"orbit", "legacy", "independent", "product"}
METHODS = CONTROLLERS | {"frozen", "strict", "tuned"}


def latency_statistics(durations_ms, deadline_ms, successful=None):
    """Retain late completions, with separately labeled deadline-met quantiles."""
    durations = np.asarray(durations_ms, dtype=float)
    if durations.ndim != 1 or not np.isfinite(durations).all() or np.any(durations < 0):
        raise ValueError("Latency samples must be finite, nonnegative milliseconds")
    if not np.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError("deadline_ms must be finite and positive")
    success = np.ones(len(durations), bool) if successful is None else np.asarray(successful, bool)
    if success.shape != durations.shape:
        raise ValueError("One success flag is required per latency sample")

    def quantiles(values):
        return {"samples": int(len(values)),
                **{f"p{percentile}_ms": float(np.percentile(values, percentile)) if len(values) else None
                   for percentile in [50, 95, 99]}}
    late = durations > deadline_ms
    return {"samples": len(durations), "deadline_ms": float(deadline_ms),
            "deadline_misses": int(late.sum()),
            "timeout_fraction": float(late.mean()) if len(durations) else None,
            "execution_failures": int((~success).sum()),
            "all_completed": quantiles(durations),
            "successful_within_deadline": quantiles(durations[success & ~late]),
            "inclusion_policy": "all completed measured rounds retained, including deadline misses; warmup excluded"}


def validate_profile_provenance(data, payload, provenance, checkpoint_hash, feature_dim):
    required = {"dataset", "dataset_sha256", "checkpoint_sha256", "backbone", "feature_dim", "counts"}
    if not required.issubset(provenance):
        raise ValueError("Fitted interface lacks cache provenance; refit with the requested checkpoint")
    if (payload.get("dataset_name") != data.name
            or payload.get("dataset_shape") != list(data.values.shape)
            or not data.metadata.get("sha256")
            or payload.get("dataset_sha256") != data.metadata["sha256"]):
        raise ValueError("Checkpoint dataset provenance does not match supplied data")
    expected = {"dataset": data.name, "dataset_sha256": data.metadata["sha256"],
                "checkpoint_sha256": checkpoint_hash, "backbone": payload.get("backbone"),
                "feature_dim": feature_dim}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("Fitted cache/checkpoint/feature provenance does not match profiling inputs")
    counts = provenance["counts"]
    if not isinstance(counts, dict) or any(not isinstance(counts.get(split), int) or counts[split] < 1
                                           for split in ["train", "val", "test"]):
        raise ValueError("Fitted cache provenance must contain positive split origin counts")


class _TimedMetadataSimulator(MetadataSimulator):
    """Measure actual validation/restriction independently of queue dispatch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accept_timings = []

    def accept(self, answer):
        start = time.perf_counter()
        accepted = False
        try:
            accepted = super().accept(answer)
            return accepted
        finally:
            end = time.perf_counter()
            self._accept_timings.append({"start_perf_seconds": start, "end_perf_seconds": end,
                                         "duration_ms": 1000*(end-start), "accepted": bool(accepted),
                                         "phase": answer.get("phase", "unspecified"),
                                         "node": answer.get("node"), "round": self.current_round})

    def drain_accept_timings(self):
        results, self._accept_timings = self._accept_timings, []
        return results


def _repair_readout(embedded, previous, scale, domains, method):
    """Correct the present aggregate readout at the previously issued gates."""
    if previous is None:
        raise ValueError("Repair requires an initialized retained-state readout")
    problem = CatalogProblem(embedded, previous["W"], previous["G"], scale, domains)
    gates = previous["gates"]
    if method == "orbit":
        center = problem.support(gates).center
    else:
        # Comparison controllers use independently centered projected families.
        center = np.zeros_like(scale)
        for j in range(problem.J):
            unit = np.zeros(problem.J)
            unit[j] = 1
            center += gates[j]*problem.support(unit).center
    return previous["anchor"] + scale*center


class _MemorySampler:
    def __init__(self, interval=.005):
        self.interval = interval
        self.process = psutil.Process()
        self.peak = int(self.process.memory_info().rss)
        self.samples = 1
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.wait(self.interval):
            self.peak = max(self.peak, int(self.process.memory_info().rss))
            self.samples += 1

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, int(self.process.memory_info().rss))
        self.samples += 1


@contextmanager
def _stage(record, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        record["stages"].append({"name": name, "start_perf_seconds": start,
                                 "end_perf_seconds": end, "duration_ms": 1000*(end-start)})


def profile_stream(data, checkpoint, interface_path, output, method="orbit", rounds=100,
                   warmup=10, deadline_ms=1000, device="cpu"):
    """Profile one real forecast at a time, including graph/feature/controller work.

    Startup data preparation/checkpoint loading is reported separately. The first
    rolling graph is initialized and populated inside round zero's charged span.
    Warmup forecasts advance the same causal stream and are retained in records;
    only their latency samples are excluded from final measured quantiles.
    Deadline misses finish and remain in the all-completed latency distribution.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown profiling method: {method}")
    if (not isinstance(rounds, int) or rounds < 1 or not isinstance(warmup, int) or warmup < 0
            or not np.isfinite(deadline_ms) or deadline_ms <= 0):
        raise ValueError("Require positive rounds/deadline and nonnegative integer warmup")
    selected = resolve_device(device)
    capability = solver_thread_capability()
    if method in CONTROLLERS and (not capability.get("thread_controlled") or capability.get("threads") != 1):
        raise RuntimeError("Profiling requires verified one-thread LP control; install the optional highspy backend")
    available = data.origins("test")
    if len(available) < rounds+warmup:
        raise ValueError("Test partition has fewer origins than warmup + requested measured rounds")
    origins = available[:rounds+warmup]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    rows, repair_rows, manifests = [], [], []
    sampler = _MemorySampler()
    sampler.start()
    started = time.perf_counter()
    checkpoint_hash = interface_hash = None
    error, bank, config, payload = None, None, None, {}
    provenance = {}
    ledger = ForecastLedger()
    gpu_peak_allocated = gpu_peak_reserved = 0
    cuda_peak_started = False

    def synchronize():
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)

    def repair_batch(record, accept_timings, simulator, previous, embedded, scale):
        if not accept_timings:
            return
        record["metadata_accept_events"].extend(accept_timings)
        accepted = [event for event in accept_timings if event["accepted"]]
        if not accepted:
            return
        if previous is None:
            # No invented zero-time repair sample without initialized state.
            record["uninitialized_repair_batches_skipped"] += 1
            return
        begin = time.perf_counter()
        repaired = _repair_readout(embedded, previous, scale, simulator.domains, method)
        finish = time.perf_counter()
        if not np.isfinite(repaired).all():
            raise FloatingPointError("Nonfinite repaired readout")
        event = {"round": record["round"], "warmup": record["warmup"],
                 "phase": accepted[0]["phase"], "accepted_answers": len(accepted),
                 "validation_and_restriction_ms": sum(e["duration_ms"] for e in accept_timings),
                 "readout_start_perf_seconds": begin, "readout_end_perf_seconds": finish,
                 "corrected_readout_ms": 1000*(finish-begin),
                 "max_correction_magnitude": float(np.max(np.abs(repaired-previous["anchor"]))),
                 "scope": "sum of measured evidence validation/domain restriction and corrected retained-state readout; queue wait/dispatch excluded"}
        event["repair_only_ms"] = event["validation_and_restriction_ms"] + event["corrected_readout_ms"]
        repair_rows.append(event)
        # The corrected present readout is intentionally not assigned to ledger.

    try:
        checkpoint_hash = file_sha256(checkpoint)
        interface_hash = file_sha256(interface_path)
        torch.set_num_threads(8)
        with threadpool_limits(limits=8, user_api="blas"):
            model, scaler, payload = load_checkpoint(checkpoint, str(selected))
            interface, archive_v, current_v, config = read_interface(interface_path)
            provenance = interface.get("provenance", {})
            validate_profile_provenance(data, payload, provenance, checkpoint_hash, model.feature_dim)
            if interface["transform"].mean.shape != (model.feature_dim,):
                raise ValueError("Fitted feature transform does not match checkpoint representation width")
            if config.epoch_rounds < 1:
                raise ValueError("Profile epoch_rounds must be positive")
            scale = np.asarray(interface["scale"], dtype=float)
            if scale.shape != (data.num_nodes, data.output_steps) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Fitted output scale is invalid for checkpoint/data")
            windows = WindowDataset(data, "test", scaler=scaler, origins=origins)
            if method != "frozen":
                projections = current_v if method == "tuned" else archive_v
                expected_projection = (4, interface["transform"].directions.shape[1]+1, data.output_steps)
                if projections.shape != expected_projection:
                    raise ValueError("Fitted projections do not match the four-archive feature/output interface")
                bank = HistoryBank(data.num_nodes, projections, config.gamma)
            caps = np.full(4, .25)
            embedded = None
            if method in CONTROLLERS:
                if interface["sources"].shape != (4, data.num_nodes, data.num_nodes):
                    raise ValueError("Profile expects the paper's four fitted source operators")
                embedded = np.zeros((4, data.num_nodes+4, data.num_nodes+4))
                embedded[:, :data.num_nodes, :data.num_nodes] = interface["sources"]
            delay = (4 if data.name == "knowair" else 12) if config.delay is None else config.delay
            refill = (.5 if data.name == "knowair" else 1/6) if config.refill is None else config.refill
            graph_window = 240 if data.name == "knowair" else 2016
            rolling, cursor, simulator, previous = None, None, None, None
            synchronize()
            if selected.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected)
                cuda_peak_started = True
            startup_ms = 1000*(time.perf_counter()-started)
            with torch.no_grad():
                for t, origin in enumerate(origins):
                    record = {"round": t, "origin": int(origin), "warmup": t < warmup,
                              "epoch": t//config.epoch_rounds, "stages": [],
                              "metadata_accept_events": [], "uninitialized_repair_batches_skipped": 0,
                              "completed": False, "successful": False}
                    synchronize()
                    tick = time.perf_counter()
                    record["start_perf_seconds"] = tick
                    try:
                        with _stage(record, "epoch_and_metadata_returns"):
                            if t % config.epoch_rounds == 0:
                                previous = None
                                if bank is not None:
                                    bank.reset()
                                if method in CONTROLLERS:
                                    manifest = generate_manifest(data.num_nodes, data.num_nodes+4,
                                        min(config.unresolved, data.num_nodes), config.domain_size,
                                        config.generator_seed+t//config.epoch_rounds, epoch=t//config.epoch_rounds)
                                    balance = simulator.balance if simulator else 0.
                                    cursor_query = simulator.cursor if simulator else 0
                                    simulator = _TimedMetadataSimulator(manifest, delay, refill, config.capacity)
                                    simulator.balance, simulator.cursor = balance, cursor_query
                                    manifests.append(manifest)
                            if simulator is not None:
                                simulator.before_round(t)
                                repair_batch(record, simulator.drain_accept_timings(), simulator,
                                             previous, embedded, scale)
                        graph = None
                        if bank is not None:
                            with _stage(record, "observation_ingestion_and_graph"):
                                if rolling is None:
                                    rolling = RollingGraph(graph_window, data.num_nodes, config.k)
                                    cursor = max(0, int(origin)-graph_window)
                                while cursor < origin:
                                    rolling.update(data.values[cursor], data.valid[cursor])
                                    cursor += 1
                                graph = rolling.operator()
                        with _stage(record, "window_and_backbone"):
                            batch = windows[t]
                            inputs = batch["x"].unsqueeze(0).to(selected)
                            if bank is not None:
                                normalized, representation = model(inputs, return_features=True)
                                realized_features = representation[0].detach().cpu().numpy().astype(np.float64)
                            else:
                                normalized = model(inputs)
                            base = (normalized*scaler.std+scaler.mean)[0].detach().cpu().numpy().astype(np.float64)
                            synchronize()
                        result, problem = None, None
                        if bank is not None:
                            with _stage(record, "frozen_transform_history_and_anchor"):
                                features = interface["transform"].transform(realized_features)
                                bank.update(features)
                                W = bank.projected()
                                anchor = base + np.einsum("nv,jvd,j->nd", graph, W, caps)
                            prediction = anchor
                        else:
                            prediction = base
                        if method in CONTROLLERS:
                            with _stage(record, "catalog_assembly_and_admission"):
                                problem = CatalogProblem(embedded, W, graph, scale, simulator.domains)
                                result = (solve_joint(problem, caps, config.epsilon,
                                                      config.max_iterations, config.time_limit)
                                          if method == "orbit" else solve_baseline(problem, caps, config.epsilon, method))
                                prediction = anchor+scale*result.center
                                previous = {"W": W.copy(), "G": graph.copy(),
                                            "anchor": anchor.copy(), "gates": result.gates.copy()}
                                record["controller_status"] = result.status
                                record["admitted_mass"] = result.mass
                                record["radius"] = result.radius
                        with _stage(record, "issuance_queue_and_dispatch"):
                            if not np.isfinite(prediction).all():
                                raise FloatingPointError("Nonfinite forecast")
                            ledger.seal(t, prediction, {"epoch": record["epoch"], "method": method})
                            record["issued_perf_seconds"] = time.perf_counter()
                            if simulator is not None:
                                requests = simulator.after_issuance(t)
                                record["requests"] = requests
                                repair_batch(record, simulator.drain_accept_timings(), simulator,
                                             previous, embedded, scale)
                        record["completed"] = True
                        record["successful"] = result is None or result.status not in {"solver_failure", "numerical_stall"}
                    except BaseException as failure:
                        record["failure"] = f"{type(failure).__name__}: {failure}"
                        raise
                    finally:
                        # Synchronize even when inference or a later stage fails.
                        synchronize()
                        record["end_perf_seconds"] = time.perf_counter()
                        record["duration_ms"] = 1000*(record["end_perf_seconds"]-tick)
                        record["deadline_ms"] = float(deadline_ms)
                        record["deadline_missed"] = record["duration_ms"] > deadline_ms
                        rows.append(record)
                    # Offline audit is after immutable issuance and outside the timer.
                    if problem is not None:
                        truth_response = problem.coefficients(simulator.manifest["truth"])@result.gates
                        record["post_issuance_discrepancy_audit"] = float(np.max(np.abs(result.center-truth_response)))
            synchronize()
            if selected.type == "cuda":
                gpu_peak_allocated = int(torch.cuda.max_memory_allocated(selected))
                gpu_peak_reserved = int(torch.cuda.max_memory_reserved(selected))
    except BaseException as failure:
        error = failure
    finally:
        try:
            synchronize()
            if cuda_peak_started:
                gpu_peak_allocated = int(torch.cuda.max_memory_allocated(selected))
                gpu_peak_reserved = int(torch.cuda.max_memory_reserved(selected))
        except BaseException as synchronization_error:
            if error is None:
                error = synchronization_error
        finally:
            sampler.stop()
            torch.set_num_threads(previous_threads)

    completed = [row for row in rows if not row["warmup"] and row["completed"]]
    measured_repairs = [row for row in repair_rows if not row["warmup"]]
    report = {"dataset": data.name, "method": method, "requested_rounds": rounds, "warmup_rounds": warmup,
              "device": str(selected), "batch_size": 1, "sample_scope": "one full-node graph per origin",
              "status": "completed" if error is None else "aborted",
              "error": None if error is None else f"{type(error).__name__}: {error}",
              "timing": latency_statistics([r["duration_ms"] for r in completed], deadline_ms,
                                            [r["successful"] for r in completed]),
              "incomplete_measured_rounds": sum(not r["warmup"] and not r["completed"] for r in rows),
              "repair_timing": latency_statistics([r["repair_only_ms"] for r in measured_repairs], deadline_ms),
              "repair_samples": measured_repairs, "all_repair_events": repair_rows,
              "round_records": rows, "startup_ms": locals().get("startup_ms"),
              "initial_graph_initialization_charged_to_round": 0 if method != "frozen" else None,
              "initial_graph_round_is_warmup": warmup > 0 and method != "frozen",
              "trace_payload_bytes": bank.payload_bytes if bank is not None else 0,
              "sampled_rss_peak_bytes": sampler.peak, "rss_samples": sampler.samples,
              "rss_sample_interval_ms": 1000*sampler.interval,
              "cuda_peak_allocated_bytes": gpu_peak_allocated, "cuda_peak_reserved_bytes": gpu_peak_reserved,
              "memory_scope": "process/model/interface setup, warmup and measurement; sampled RSS can miss sub-interval peaks",
              "cuda_memory_scope": "after checkpoint/interface setup, through warmup and measurement",
              "thread_settings": {"torch": 8, "blas": 8, "matching": "single-thread SciPy assignment",
                                  "lp": capability},
              "provenance": {"dataset_sha256": data.metadata.get("sha256"),
                             "checkpoint_sha256": checkpoint_hash, "fitted_cache": provenance,
                             "checkpoint_path": str(Path(checkpoint).resolve()),
                             "interface_path": str(Path(interface_path).resolve()),
                             "interface_sha256": interface_hash,
                             "checkpoint_training_limited": payload.get("run_config", {}).get("limited"),
                             "platform": platform.platform(), "python": platform.python_version()},
              "config": asdict(config) if config is not None else None,
              "formal_interval_certificate": False,
              "timing_scope": "per-round graph ingestion/build, on-demand backbone/features, histories, admission, issuance and dispatch; no cached inference",
              "excluded": ["startup checkpoint loading and whole-series causal preprocessing (reported separately)",
                           "post-issuance truth audit and final artifact serialization", "exogenous metadata waiting"],
              "deadline_policy": "complete each forecast; mark wall-time misses; solver time limit is separately configured and soft",
              "host_exclusivity": "must be ensured by operator; not independently verified",
              "metadata_policy_active": method in CONTROLLERS,
              "current_comparator_qualification": (
                  "Uses current_v and config from the supplied interface; supply independently selected tuned_interface.npz for the paper tuned-current comparison"
                  if method == "tuned" else "not applicable"),
              "forecast_rewrites": 0}
    write_json(output/"profile.json", report)
    write_json(output/"manifests.json", manifests)
    if ledger.predictions:
        np.save(output/"predictions.npy", np.stack([ledger.predictions[t] for t in sorted(ledger.predictions)]))
    if error is not None:
        raise error
    return report
