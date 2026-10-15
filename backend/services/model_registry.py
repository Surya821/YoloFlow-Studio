"""Shared model registry: COCO base weights, project training runs and user-uploaded YOLO models.

Model keys:
    base:yolo11n.pt          official COCO weights (downloaded to workspace/models on first use)
    run:<run_id>             best.pt from a project's training run
    custom:<filename>        uploaded Ultralytics YOLO detection model (.pt / .onnx)
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from core import storage
from core.catalog import AUTOLABEL_BASE_MODELS
from core.storage import BadRequest, NotFound

CUSTOM_EXTS = {".pt", ".onnx"}
MAX_UPLOAD_BYTES = 2 * 1024 ** 3

_cache: dict[str, object] = {}
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def custom_dir() -> Path:
    d = storage.MODELS_DIR / "custom"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _custom_index_path() -> Path:
    return custom_dir() / "models.json"


def resolve(project: str | None, key: str) -> Path:
    kind, _, value = (key or "").partition(":")
    if kind == "base":
        if value not in AUTOLABEL_BASE_MODELS:
            raise BadRequest(f"Unknown base model '{value}'.")
        storage.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = storage.MODELS_DIR / value  # Ultralytics downloads the asset here on first use
        # An interrupted download leaves a truncated file Ultralytics would never replace;
        # .pt checkpoints are zip archives, so a missing central directory means "download again".
        # (A file written to in the last 2 minutes is probably still downloading - leave it alone.)
        if path.exists() and time.time() - path.stat().st_mtime > 120 and not zipfile.is_zipfile(path):
            evict(path)
            path.unlink(missing_ok=True)
        return path
    if kind == "run":
        if not project:
            raise BadRequest("A project is required for run models.")
        root = storage.require_project(project) / "outputs"
        path = storage.ensure_within(root, root / value / "artifacts" / "best.pt")
        if not path.exists():
            raise NotFound("That training run has no best.pt weights.")
        return path
    if kind == "custom":
        root = custom_dir()
        path = storage.ensure_within(root, root / value)
        if not path.is_file() or path.suffix.lower() not in CUSTOM_EXTS:
            raise NotFound("Uploaded model not found.")
        return path
    raise BadRequest(f"Invalid model key '{key}'.")


def load_fresh(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path))  # task comes from the checkpoint / ONNX metadata, never forced


def get_cached(path: Path):
    """Shared instance + lock for quick, short-lived predictions (annotator / single images)."""
    key = str(path)
    with _guard:
        if key not in _cache:
            _cache[key] = load_fresh(path)
            _locks[key] = threading.Lock()
        return _cache[key], _locks[key]


def evict(path: Path) -> None:
    with _guard:
        _cache.pop(str(path), None)
        _locks.pop(str(path), None)


def class_names(model) -> list[str]:
    names = model.names
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


# ---------------------------------------------------------------- listing

def list_models(project: str | None) -> list[dict]:
    models = []
    if project:
        out_dir = storage.require_project(project) / "outputs"
        for run_dir in sorted(out_dir.glob("*"), reverse=True):
            if (run_dir / "artifacts" / "best.pt").exists():
                run = storage.read_json(run_dir / "run.json", {})
                m = (run.get("metrics") or {}).get("map50")
                label = f"{run.get('model', 'run')} · {run_dir.name}" + (f" · mAP50 {m:.3f}" if m is not None else "")
                models.append({"key": f"run:{run_dir.name}", "label": label, "kind": "run",
                               "classes": run.get("classes")})
    for entry in storage.read_json(_custom_index_path(), {}).values():
        if (custom_dir() / entry["filename"]).exists():
            models.append({"key": f"custom:{entry['filename']}", "label": entry["name"], "kind": "custom",
                           "classes": entry.get("classes"), "size": entry.get("size"),
                           "format": entry.get("format"), "uploaded_at": entry.get("uploaded_at")})
    models += [{"key": f"base:{m}", "label": f"{m} (COCO)", "kind": "base"} for m in AUTOLABEL_BASE_MODELS]
    return models


# ---------------------------------------------------------------- upload / delete

def _validate_yolo(path: Path) -> dict:
    """Accept only Ultralytics YOLO detection models: they must load, report task=detect, and run."""
    import numpy as np

    try:
        model = load_fresh(path)
        task = getattr(model, "task", None)
        if task != "detect":
            raise BadRequest(f"Only YOLO detection models are supported (this model's task is '{task}').")
        model.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)  # also resolves ONNX metadata
        names = class_names(model)
    except BadRequest:
        raise
    except Exception as exc:  # noqa: BLE001
        raise BadRequest(f"Not a valid Ultralytics YOLO model: {str(exc)[:300]}") from exc
    if not names:
        raise BadRequest("Model has no class names - export it with Ultralytics so metadata is embedded.")
    if path.suffix.lower() == ".onnx" and names == [str(i) for i in range(len(names))]:
        raise BadRequest("ONNX file has no Ultralytics metadata - export it with `model.export(format='onnx')`.")
    return {"classes": names}


async def save_upload(upload, display_name: str | None = None) -> dict:
    original = upload.filename or "model.pt"
    ext = os.path.splitext(original)[1].lower()
    if ext not in CUSTOM_EXTS:
        raise BadRequest("Only YOLO model files (.pt or .onnx) can be uploaded.")
    tmp_dir = Path(tempfile.mkdtemp(prefix="yf_model_", dir=custom_dir()))
    tmp = tmp_dir / f"upload{ext}"
    size = 0
    try:
        with open(tmp, "wb") as f:
            while chunk := await upload.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise BadRequest("Model file is larger than 2 GB.")
                f.write(chunk)
        import asyncio

        info = await asyncio.to_thread(_validate_yolo, tmp)
        dest = storage.unique_path(custom_dir(), storage.safe_filename(original))
        shutil.move(str(tmp), dest)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    entry = {
        "filename": dest.name,
        "name": (display_name or "").strip() or dest.stem,
        "format": ext.lstrip("."),
        "size": size,
        "classes": info["classes"],
        "uploaded_at": time.time(),
    }

    with storage.lock_for("custom-models"):
        index = storage.read_json(_custom_index_path(), {})
        index[dest.name] = entry
        storage.write_json(_custom_index_path(), index)
    return {**entry, "key": f"custom:{dest.name}"}


def delete_custom(filename: str) -> None:
    path = resolve(None, f"custom:{filename}")
    evict(path)
    path.unlink(missing_ok=True)
    with storage.lock_for("custom-models"):
        index = storage.read_json(_custom_index_path(), {})
        index.pop(filename, None)
        storage.write_json(_custom_index_path(), index)
