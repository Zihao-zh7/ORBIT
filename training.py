"""Reproducible backbone fitting and immutable frozen-feature caches.

The paper supplies no checkpoints or optimizer configuration. Defaults reconstruct
the traffic authors' configurations; their use on KnowAir is an explicit assumption.
Every limited run records its limits, and no validation/test targets update weights.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbones import make_backbone
from .data import Standardizer, TimeSeriesData, WindowDataset, file_sha256


INPUT_CONVENTION = "target+time_of_day+day_of_week"


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch installation has no available CUDA device")
    return torch.device(device)


def _limit(origins: np.ndarray, maximum: int | None) -> np.ndarray:
    if maximum is None:
        return origins
    if maximum < 1:
        raise ValueError("origin limit must be positive")
    return origins[:maximum]


def _loader(dataset, batch_size, device, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda", drop_last=False)


def _native_prediction(model, batch, scaler, device):
    x = batch["x"].to(device, non_blocking=True)
    return model(x) * scaler.std + scaler.mean


def _validation_mae(model, loader, scaler, device):
    model.eval()
    error, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            prediction = _native_prediction(model, batch, scaler, device)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            error += float((prediction - target).abs().masked_select(valid).sum().item())
            count += int(valid.sum().item())
    if not count:
        raise ValueError("validation origins contain no valid targets")
    return error / count


def train_backbone(data: TimeSeriesData, output: str | Path, backbone: str = "stid",
                   seed: int = 0, epochs: int = 100, batch_size: int = 64,
                   device: str = "auto", max_train_origins: int | None = None,
                   max_val_origins: int | None = None, identity_free: bool = False,
                   model_kwargs: dict | None = None, patience: int = 20,
                   learning_rate: float | None = None, weight_decay: float | None = None,
                   num_threads: int = 4) -> Path:
    """Train on complete training targets, select a checkpoint by validation MAE.

    Loss and validation metric are masked MAE in native target units over all leads.
    max_*_origins select chronological prefixes for explicitly labeled smoke runs.
    """
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    output = Path(output)
    checkpoint = output if output.suffix == ".pt" else output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    selected_device = resolve_device(device)
    scaler = Standardizer.fit(data)
    train_data = WindowDataset(data, "train", scaler=scaler,
                               origins=_limit(data.origins("train"), max_train_origins))
    val_data = WindowDataset(data, "val", scaler=scaler, prepared=train_data.prepared,
                             origins=_limit(data.origins("val"), max_val_origins))
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("training and validation both need complete forecast windows")
    train_loader = _loader(train_data, batch_size, selected_device, shuffle=True)
    val_loader = _loader(val_data, batch_size, selected_device)
    model_config = {"num_nodes": data.num_nodes, "input_steps": data.input_steps,
                    "output_steps": data.output_steps, "slots_per_day": data.slots_per_day,
                    "identity_free": identity_free}
    model_config.update(model_kwargs or {})
    backbone = backbone.lower()
    model = make_backbone(backbone, **model_config).to(selected_device)
    lr = learning_rate if learning_rate is not None else (0.002 if backbone == "stid" else 0.001)
    decay = weight_decay if weight_decay is not None else (0.0003 if backbone == "staeformer" and data.name == "metr-la" else 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    milestones = [1, 50, 80] if backbone == "stid" else ([10, 30] if data.name == "pems-bay" else [20, 30])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones,
                                                    gamma=0.5 if backbone == "stid" else 0.1)
    run_config = {
        "backbone": backbone, "seed": seed, "epochs_requested": epochs,
        "batch_size": batch_size, "device": str(selected_device), "patience": patience,
        "learning_rate": lr, "weight_decay": decay, "milestones": milestones,
        "max_train_origins": max_train_origins, "max_val_origins": max_val_origins,
        "train_origins": len(train_data), "val_origins": len(val_data),
        "limited": max_train_origins is not None or max_val_origins is not None,
        "input_convention": INPUT_CONVENTION,
        "assumptions": ["Original ORBIT optimizer configuration and checkpoints not provided",
                        "Traffic architecture/optimizer reconstructed from official STID/STAEformer repositories",
                        "KnowAir uses target and calendar channels with the traffic backbone layout",
                        "Early stopping and train-only valid-target normalization are reconstructed choices",
                        "Invalid input forward fill uses prior observations then training-only node mean"],
        "torch_version": str(torch.__version__), "model_config": model_config,
    }
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total_error, total_count = 0.0, 0
        for batch in train_loader:
            target = batch["target"].to(selected_device, non_blocking=True)
            valid = batch["valid"].to(selected_device, non_blocking=True)
            count = int(valid.sum().item())
            if count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = _native_prediction(model, batch, scaler, selected_device)
            absolute = (prediction - target).abs().masked_select(valid)
            loss = absolute.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite backbone loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_error += float(absolute.detach().sum().item())
            total_count += count
        if total_count == 0:
            raise ValueError("selected training origins contain no valid targets")
        val_mae = _validation_mae(model, val_loader, scaler, selected_device)
        scheduler.step()
        row = {"epoch": epoch, "train_mae": total_error / total_count,
               "validation_mae": val_mae, "seconds": time.perf_counter() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_mae < best:
            best, stale = val_mae, 0
            payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       "backbone": backbone, "model_config": model_config, "scaler": scaler.as_dict(),
                       "dataset_name": data.name, "dataset_shape": list(data.values.shape),
                       "dataset_sha256": data.metadata.get("sha256"), "data_metadata": data.metadata,
                       "best_epoch": epoch, "best_validation_mae": best,
                       "run_config": run_config}
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
        else:
            stale += 1
        (checkpoint.parent / "training.json").write_text(
            json.dumps({"config": run_config, "history": history,
                        "best_validation_mae": best, "epochs_completed": epoch}, indent=2), encoding="utf-8")
        if stale >= patience:
            break
    (checkpoint.parent / 'training.json').write_text(
        json.dumps({'config':run_config,'history':history,'best_validation_mae':best,
                    'epochs_completed':len(history),'completed':True,
                    'stopping_reason':'early_stopping' if stale>=patience else 'epoch_limit'},indent=2),encoding='utf-8')
    return checkpoint


def load_checkpoint(checkpoint: str | Path, device: str = "auto"):
    """Load a checkpoint produced by this module (trusted local files only)."""
    selected = resolve_device(device)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.12 compatibility.
        payload = torch.load(checkpoint, map_location="cpu")
    model = make_backbone(payload["backbone"], **payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(selected).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = payload["scaler"]
    scaler = Standardizer(params["mean"], params["std"], np.asarray(params["node_fill"]))
    return model, scaler, payload


def extract_cache(data: TimeSeriesData, checkpoint: str | Path, output: str | Path,
                  batch_size: int = 64, device: str = "auto",
                  max_origins: int | dict[str, int] | None = None, *,
                  splits=('train', 'val', 'test'), feature_dtype='float64') -> dict:
    """Write native predictions/targets/masks and pre-head features per split.

    Returns read-only .npy memory maps for the requested splits. Float32 storage
    preserves the realized FP32 eval-mode stream exactly; residual operations
    promote it to FP64 on use. The default retains the original FP64 cache API.
    Separate phase directories allow training/validation caches to be released
    before test extraction. No expanded feature arrays live in RAM.
    """
    from .storage import _cache_layouts, _cache_request
    if (isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError('Cache batch size must be a positive integer')
    requested, feature_dtype, split_origins = _cache_request(data, splits, feature_dtype, max_origins)
    output = Path(output)
    selected = resolve_device(device)
    model, scaler, payload = load_checkpoint(checkpoint, str(selected))
    if payload["dataset_name"] != data.name or payload["dataset_shape"] != list(data.values.shape):
        raise ValueError("checkpoint belongs to another dataset or node count")
    expected_hash = payload.get("dataset_sha256")
    if expected_hash and expected_hash != data.metadata.get("sha256"):
        raise ValueError("dataset fingerprint differs from training checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate an older completion record before any existing arrays open w+.
    (output/'metadata.json').write_text(json.dumps({'status':'in_progress','completed':False,
        'dataset':data.name,'checkpoint':str(Path(checkpoint).resolve())}),encoding='utf-8')
    result, counts, prepared = {}, {}, None
    for split in requested:
        origins = split_origins[split]
        windows = WindowDataset(data, split, scaler=scaler, origins=origins, prepared=prepared)
        prepared = windows.prepared
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        layouts = _cache_layouts(len(origins), data.num_nodes, data.output_steps,
                                 model.feature_dim, feature_dtype)
        maps = {key: np.lib.format.open_memmap(directory / (key + ".npy"), mode="w+", dtype=dtype, shape=dimensions)
                for key, (dimensions, dtype) in layouts.items()}
        cursor = 0
        with torch.no_grad():
            for batch in _loader(windows, batch_size, selected):
                prediction, features = model(batch["x"].to(selected, non_blocking=True), return_features=True)
                n = len(prediction)
                sl = slice(cursor, cursor + n)
                maps["predictions"][sl] = (prediction * scaler.std + scaler.mean).cpu().numpy()
                maps["features"][sl] = features.cpu().numpy()
                maps["origins"][sl] = batch["origin"].numpy()
                maps["targets"][sl] = batch["target"].numpy()
                maps["valid"][sl] = batch["valid"].numpy()
                cursor += n
        for mapped in maps.values():
            mapped.flush()
        del maps
        result[split] = {key: np.load(directory / (key + ".npy"), mmap_mode="r") for key in layouts}
        counts[split] = len(origins)
    metadata = {"status":"complete", "completed":True,"dataset": data.name, "dataset_sha256": data.metadata.get("sha256"),
                "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_sha256(checkpoint),
                "backbone": payload["backbone"], "feature_dim": model.feature_dim,
                "feature_dtype": feature_dtype.name, "requested_splits": list(requested),
                "counts": counts, "max_origins": max_origins,
                "limited": max_origins is not None or payload["run_config"]["limited"],
                "input_convention": INPUT_CONVENTION, "training_config": payload["run_config"]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result
