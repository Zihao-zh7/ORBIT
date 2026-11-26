"""Raw dataset loading, chronological partitions and causal backbone windows.

KnowAir schema: https://github.com/shuowang-ai/PM2.5-GNN/blob/main/dataset.py
and config.yaml. Last channel is PM2.5; dates are GMT, 2015--2018.
Meteorology is preserved, but the reconstructed backbones use target + calendar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TimeSeriesData:
    name: str
    values: np.ndarray
    valid: np.ndarray
    timestamps: np.ndarray
    splits: dict[str, tuple[int, int]]
    input_steps: int
    output_steps: int
    leads: tuple[int, ...]
    slots_per_day: int
    covariates: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.values = np.asarray(self.values)
        self.valid = np.asarray(self.valid, dtype=bool) & np.isfinite(self.values)
        self.timestamps = np.asarray(self.timestamps, dtype="datetime64[ns]")
        if self.values.ndim != 2 or self.valid.shape != self.values.shape:
            raise ValueError("values and validity must both have shape [time,node]")
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamp count differs from data length")
        if self.input_steps < 1 or self.output_steps < 1:
            raise ValueError("positive window sizes required")
        previous = 0
        for split in ("train", "val", "test"):
            start, end = self.splits[split]
            if start != previous or not start < end <= len(self.values):
                raise ValueError("partitions must be ordered, contiguous and nonempty")
            previous = end
        if previous != len(self.values):
            raise ValueError("partitions must cover all observations")

    def origins(self, split: str) -> np.ndarray:
        """Origin t predicts x[t:t+d]; input is strictly x[t-L:t]."""
        start, end = self.splits[split]
        return np.arange(max(start, self.input_steps), end - self.output_steps + 1,
                         dtype=np.int64)

    @property
    def num_nodes(self) -> int:
        return self.values.shape[1]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(root: str | Path, name: str) -> TimeSeriesData:
    """Load supplied files without interpolation or modifying original data.

    Traffic ratios use floor(0.7*T), floor(0.1*T), remainder. PEMS daylight
    saving jump remains in the calendar index; a model step is one raw row.
    KnowAir zeros are retained as valid because its original missingness mask
    was not supplied. This is an explicit reconstruction assumption.
    """
    canonical = name.lower().replace("_", "-")
    canonical = {"metrla": "metr-la", "pemsbay": "pems-bay", "know-air": "knowair"}.get(canonical, canonical)
    filename = {"metr-la": "metr-la.h5", "pems-bay": "pems-bay.h5", "knowair": "KnowAir.npy"}.get(canonical)
    if filename is None:
        raise ValueError(f"unknown dataset {name!r}")
    root = Path(root)
    path = root / filename
    if not path.exists() and (root / "data" / filename).exists():
        path = root / "data" / filename
    if canonical == "knowair":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.shape != (11688, 184, 18):
            raise ValueError(f"expected original KnowAir [11688,184,18], received {raw.shape}")
        values = raw[..., -1]
        valid = np.isfinite(values) & (values >= 0)
        covariates = raw[..., :-1]
        timestamps = pd.date_range("2015-01-01", periods=len(raw), freq="3h").to_numpy()
        cuts = (0, 5848, 8768, 11688)
        input_steps = output_steps = 8
        leads, slots_per_day = (2, 4, 8), 8
        node_ids = [str(i) for i in range(raw.shape[1])]
        assumptions = ["KnowAir timestamps reconstructed from official GMT data_start and 3-hour cadence",
                       "KnowAir zeros retained; original validity mask unavailable",
                       "KnowAir node indices retained because station metadata was not supplied"]
    else:
        # Direct fixed-format HDF read avoids old pandas timezone/frequency metadata.
        with h5py.File(path, "r") as handle:
            key = "df" if canonical == "metr-la" else "speed"
            group = handle[key]
            values = group["block0_values"][:]
            timestamps = group["axis1"][:].astype("datetime64[ns]")
            node_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in group["axis0"][:]]
        valid = np.isfinite(values) & (values > 0)
        covariates = None
        count = len(values)
        n_train, n_val = int(count * 0.7), int(count * 0.1)
        cuts = (0, n_train, n_train + n_val, count)
        input_steps = output_steps = 12
        leads, slots_per_day = (3, 6, 12), 288
        assumptions = ["Traffic zero/nonfinite readings excluded from target metrics and graph correlations",
                       "Traffic split rounding: floor(70% rows), floor(10% rows), remainder",
                       "Sampling rounds retain original rows including the PEMS-BAY daylight-saving jump"]
    splits = dict(zip(("train", "val", "test"), zip(cuts[:-1], cuts[1:])))
    interval = np.timedelta64(24 * 60 // slots_per_day, "m")
    irregular = np.flatnonzero(np.diff(timestamps) != interval)
    metadata = {
        "path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size,
        "shape": list(values.shape), "dtype": str(values.dtype), "node_ids": node_ids,
        "start": str(timestamps[0]), "end": str(timestamps[-1]),
        "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()),
        "zero_count": int((values == 0).sum()), "nonfinite_count": int((~np.isfinite(values)).sum()),
        "timestamp_jumps": [{"before": str(timestamps[i]), "after": str(timestamps[i + 1])}
                            for i in irregular],
        "splits": {k: list(v) for k, v in splits.items()}, "assumptions": assumptions,
        "input_convention": "target+time_of_day+day_of_week",
        "covariates_used": False, "physical_graph_available": False,
    }
    return TimeSeriesData(canonical, values, valid, timestamps, splits, input_steps,
                          output_steps, leads, slots_per_day, covariates, metadata)


@dataclass
class Standardizer:
    mean: float
    std: float
    node_fill: np.ndarray

    @classmethod
    def fit(cls, data: TimeSeriesData) -> "Standardizer":
        start, end = data.splits["train"]
        values, valid = data.values[start:end], data.valid[start:end]
        observed = values[valid]
        if observed.size == 0:
            raise ValueError("no valid training targets")
        mean, std = float(observed.mean()), max(float(observed.std()), 1e-6)
        counts = valid.sum(axis=0)
        sums = np.where(valid, values, 0).sum(axis=0)
        fill = np.divide(sums, counts, out=np.full(data.num_nodes, mean), where=counts > 0)
        return cls(mean, std, fill)

    def as_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "node_fill": self.node_fill.tolist()}


def causal_fill(values: np.ndarray, valid: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Forward fill only; fallback is fixed from the training partition."""
    filled = np.empty(values.shape, dtype=np.float32)
    last = np.array(fallback, dtype=np.float64, copy=True)
    for t in range(len(values)):
        last = np.where(valid[t], values[t], last)
        filled[t] = last
    return filled


class WindowDataset(Dataset):
    """Lazy windows, no expanded [all_origins,window,nodes] allocation."""
    def __init__(self, data: TimeSeriesData, split: str, scaler: Standardizer | None = None,
                 origins: np.ndarray | None = None, prepared: np.ndarray | None = None):
        self.data = data
        self.scaler = scaler or Standardizer.fit(data)
        self.origins = data.origins(split) if origins is None else np.asarray(origins, dtype=np.int64)
        allowed = data.origins(split)
        if not np.isin(self.origins, allowed).all():
            raise ValueError("an origin has inputs or targets outside its permitted interval")
        if prepared is None:
            filled = causal_fill(data.values, data.valid, self.scaler.node_fill)
            self.prepared = (filled - self.scaler.mean) / self.scaler.std
        else:
            self.prepared = prepared
        dates = pd.DatetimeIndex(data.timestamps)
        self.tod = np.asarray((dates.hour * 60 + dates.minute) / 1440, dtype=np.float32)
        self.dow = np.asarray(dates.dayofweek, dtype=np.float32)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        t = int(self.origins[index])
        lo = t - self.data.input_steps
        x = np.empty((self.data.input_steps, self.data.num_nodes, 3), dtype=np.float32)
        x[..., 0] = self.prepared[lo:t]
        x[..., 1] = self.tod[lo:t, None]
        x[..., 2] = self.dow[lo:t, None]
        target = self.data.values[t:t + self.data.output_steps].T
        valid = self.data.valid[t:t + self.data.output_steps].T
        return {"x": torch.from_numpy(x),
                "target": torch.from_numpy(np.where(valid, target, 0).astype(np.float32)),
                "valid": torch.from_numpy(valid.copy()), "origin": t}
