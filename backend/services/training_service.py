"""Training run orchestration for Kaggle (remote GPU) and local (subprocess) backends.

Each run lives in workspace/projects/{project}/outputs/{run_id}/ with:
  run.json      - state machine + config + metrics
  train.log     - combined log
  artifacts/    - best.pt, best.onnx, results.csv/png, confusion matrices, curves
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

from core import storage
from core.catalog import MODEL_FAMILIES, OPTIMIZERS, resolve_model_name
from core.storage import BadRequest, NotFound
from services import annotation_service, kaggle_service, train_script

POLL_SECONDS = 15
ACTIVE = {"preparing", "queued", "running", "downloading"}
TERMINAL = {"completed", "error", "cancelled"}
ARTIFACT_NAMES = {"best.pt", "best.onnx", "last.pt", "results.csv", "results.png", "metrics.json",
                  "confusion_matrix.png", "confusion_matrix_normalized.png", "args.yaml", "labels.jpg"}

_procs: dict[str, subprocess.Popen] = {}
_threads: dict[str, threading.Thread] = {}


# ---------------------------------------------------------------- run records

def _runs_dir(project: str) -> Path:
    return storage.require_project(project) / "outputs"


def _run_dir(project: str, run_id: str) -> Path:
    root = _runs_dir(project)
    path = storage.ensure_within(root, root / run_id)
    if not (path / "run.json").exists():
        raise NotFound("Run not found.")
    return path


def load_run(project: str, run_id: str) -> dict:
    return storage.read_json(_run_dir(project, run_id) / "run.json", {})


def _update_run(project: str, run_id: str, **changes) -> dict:
    path = _run_dir(project, run_id) / "run.json"
    with storage.lock_for(f"run:{project}:{run_id}"):
        run = storage.read_json(path, {})
        stage = changes.get("stage")
        if stage and stage != run.get("stage"):
            run.setdefault("timeline", []).append({"stage": stage, "at": time.time()})
        run.update(changes)
        run["updated_at"] = time.time()
        storage.write_json(path, run)
        return run


def _append_log(project: str, run_id: str, text: str) -> None:
    with open(_run_dir(project, run_id) / "train.log", "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def list_runs(project: str) -> list[dict]:
    out = []
    for d in _runs_dir(project).iterdir():
        run = storage.read_json(d / "run.json")
        if run:
            out.append(run)
    return sorted(out, key=lambda r: r.get("created_at", 0), reverse=True)


def delete_run(project: str, run_id: str) -> None:
    run = load_run(project, run_id)
    if run.get("status") in ACTIVE and run.get("backend") == "local":
        cancel_run(project, run_id)
    shutil.rmtree(_run_dir(project, run_id), ignore_errors=True)


# ---------------------------------------------------------------- config validation

def normalize_config(cfg: dict) -> dict:
    defaults = storage.load_config()["training_defaults"]
    merged = {**defaults, **{k: v for k, v in (cfg or {}).items() if v is not None}}
    if not any(f["id"] == merged["family"] for f in MODEL_FAMILIES):
        raise BadRequest("Unknown model family.")
    try:
        resolve_model_name(merged["family"], merged["variant"], bool(merged["pretrained"]))
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc
    if merged["optimizer"] not in OPTIMIZERS:
        raise BadRequest("Unknown optimizer.")
    merged["epochs"] = int(merged["epochs"])
    batch = float(merged["batch"])
    merged["batch"] = batch if 0 < batch < 1 else int(batch)  # -1 = auto, 0.x = memory fraction
    merged["imgsz"] = int(merged["imgsz"])
    merged["lr0"] = float(merged["lr0"])
    merged["patience"] = int(merged.get("patience", 100))
    merged["pretrained"] = bool(merged["pretrained"])
    if merged["epochs"] < 1 or merged["imgsz"] < 32 or merged["imgsz"] % 32:
        raise BadRequest("Epochs must be >= 1 and image size a multiple of 32.")
    return merged


# ---------------------------------------------------------------- run creation

def create_run(project: str, backend: str, cfg: dict) -> dict:
    if backend not in ("kaggle", "local"):
        raise BadRequest("backend must be 'kaggle' or 'local'.")
    storage.require_project(project)
    if not storage.load_meta(project)["classes"]:
        raise BadRequest("Add at least one class before training.")
    counts = annotation_service.split_counts(project)
    if counts["train"] + counts["val"] == 0:
        raise BadRequest("No annotated images - label some frames before training.")
    if backend == "kaggle":
        kaggle_service._credentials()
        if any(r["backend"] == "kaggle" and r["status"] in ACTIVE for r in list_runs(project)):
            raise BadRequest("A Kaggle run is already active for this project (they share one kernel).")
    cfg = normalize_config(cfg)

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{backend}"
    run_dir = _runs_dir(project) / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    now = time.time()
    run = {
        "id": run_id,
        "project": project,
        "backend": backend,
        "status": "preparing",
        "stage": "created",
        "timeline": [{"stage": "created", "at": now}],
        "created_at": now,
        "updated_at": now,
        "config": cfg,
        "model": resolve_model_name(cfg["family"], cfg["variant"], cfg["pretrained"]),
        "classes": [c["name"] for c in storage.load_meta(project)["classes"]],
        "dataset_counts": counts,
        "kaggle": {},
        "progress": {"epoch": 0, "epochs": cfg["epochs"]},
        "metrics": None,
        "artifacts": [],
        "error": None,
    }
    storage.write_json(run_dir / "run.json", run)
    (run_dir / "train.log").write_text("", encoding="utf-8")
    target = _kaggle_pipeline if backend == "kaggle" else _local_pipeline
    _spawn(project, run_id, target)
    return run


def _spawn(project: str, run_id: str, target, *args) -> None:
    def wrapper():
        try:
            target(project, run_id, *args)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                _append_log(project, run_id, f"[yoloflow] ERROR: {exc}")
                _update_run(project, run_id, status="error", stage="error", error=str(exc))
            except NotFound:
                pass  # run deleted mid-flight

    t = threading.Thread(target=wrapper, daemon=True, name=f"run-{run_id}")
    _threads[run_id] = t
    t.start()


# ---------------------------------------------------------------- kaggle pipeline

def _kaggle_pipeline(project: str, run_id: str) -> None:
    run = load_run(project, run_id)
    cfg = run["config"]
    run_dir = _run_dir(project, run_id)
    staging = storage.require_project(project) / ".kaggle_staging"

    _update_run(project, run_id, stage="packaging")
    _append_log(project, run_id, "[yoloflow] packaging dataset")
    api, username = kaggle_service.get_api()
    zip_path = kaggle_service.package_dataset(project, staging / "dataset", run_id)
    size_mb = zip_path.stat().st_size / 1e6
    _append_log(project, run_id, f"[yoloflow] dataset.zip = {size_mb:.1f} MB")

    _update_run(project, run_id, stage="uploading_dataset")
    _append_log(project, run_id, "[yoloflow] uploading dataset to Kaggle")
    dataset_ref, created = kaggle_service.push_dataset(api, username, project, staging / "dataset", run_id)
    _append_log(project, run_id, f"[yoloflow] dataset {'created' if created else 'versioned'}: {dataset_ref}")

    _update_run(project, run_id, stage="processing_dataset",
                kaggle={"dataset_ref": dataset_ref, "dataset_url": f"https://www.kaggle.com/datasets/{dataset_ref}"})
    kaggle_service.wait_dataset_ready(
        api, dataset_ref,
        on_tick=lambda s: _append_log(project, run_id, f"[yoloflow] dataset status: {s}"))

    _update_run(project, run_id, stage="pushing_kernel")
    kernel_dir = staging / "kernel"
    model_source = run["model"]
    kernel_ref = kaggle_service.build_kernel(username, project, run_id, dataset_ref, cfg, model_source, kernel_dir)
    shutil.copy2(kernel_dir / "train.py", run_dir / "train.py")
    pushed = kaggle_service.push_kernel(api, kernel_dir, cfg.get("accelerator"))
    _append_log(project, run_id, f"[yoloflow] kernel pushed: {kernel_ref} (version {pushed.get('version')})")
    run = _update_run(project, run_id, status="queued", stage="queued", kaggle={
        **load_run(project, run_id)["kaggle"],
        "kernel_ref": kernel_ref,
        "kernel_url": f"https://www.kaggle.com/code/{kernel_ref}",
        "version": pushed.get("version"),
    })
    _kaggle_poll(project, run_id)


def _kaggle_poll(project: str, run_id: str) -> None:
    run = load_run(project, run_id)
    ref = run["kaggle"]["kernel_ref"]
    failures = 0
    last_log_len = 0
    time.sleep(5)
    while True:
        try:
            run = load_run(project, run_id)
        except NotFound:
            return
        if run["status"] in TERMINAL:
            return
        try:
            api, _ = kaggle_service.get_api()
            status, failure = kaggle_service.kernel_status(api, ref)
            failures = 0
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _update_run(project, run_id, poll_error=str(exc)[:300])
            if failures >= 20:
                raise
            time.sleep(POLL_SECONDS)
            continue

        changes = {"kaggle_status": status, "last_polled": time.time(), "poll_error": None}
        if status in ("QUEUED", "NEW_SCRIPT"):
            changes.update(status="queued", stage="queued")
        elif status == "RUNNING":
            changes.update(status="running", stage="running")
        elif status in ("CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"):
            _update_run(project, run_id, status="cancelled", stage="cancelled", **changes)
            return
        elif status == "ERROR":
            _update_run(project, run_id, stage="downloading", **changes)
            _kaggle_finish(project, run_id, api, ref, ok=False, failure=failure)
            return
        elif status == "COMPLETE":
            _update_run(project, run_id, status="downloading", stage="downloading", **changes)
            _kaggle_finish(project, run_id, api, ref, ok=True)
            return
        _update_run(project, run_id, **changes)

        # Best-effort live log (may be empty until the session ends).
        text = kaggle_service.kernel_log(api, ref)
        if text and len(text) > last_log_len:
            _write_remote_log(project, run_id, text)
            last_log_len = len(text)
            _update_progress_from_log(project, run_id, text)
        time.sleep(POLL_SECONDS)


def _write_remote_log(project: str, run_id: str, text: str) -> None:
    (_run_dir(project, run_id) / "remote.log").write_text(text, encoding="utf-8")


def _update_progress_from_log(project: str, run_id: str, text: str) -> None:
    hits = re.findall(r"^\s*(\d+)/(\d+)\s", text.replace("\r", "\n"), flags=re.M)
    if hits:
        epoch, total = map(int, hits[-1])
        _update_run(project, run_id, progress={"epoch": epoch, "epochs": total})


def _kaggle_finish(project: str, run_id: str, api, ref: str, ok: bool, failure: str | None = None) -> None:
    run_dir = _run_dir(project, run_id)
    download = run_dir / "kaggle_output"
    _append_log(project, run_id, "[yoloflow] downloading kernel output")
    try:
        files = kaggle_service.download_output(api, ref, download)
        _append_log(project, run_id, f"[yoloflow] downloaded {len(files)} files")
    except Exception as exc:  # noqa: BLE001
        _append_log(project, run_id, f"[yoloflow] output download failed: {exc}")
    text = kaggle_service.kaggle_log_from_download(download) or kaggle_service.kernel_log(api, ref)
    if text:
        _write_remote_log(project, run_id, text)
    _collect_artifacts(project, run_id, download)
    run = _finalize_metrics(project, run_id)
    if ok and run.get("artifacts") and "best.pt" in run["artifacts"]:
        _update_run(project, run_id, status="completed", stage="completed")
    else:
        msg = failure or ("Kernel completed but best.pt was not produced - see the log." if ok
                          else "Kaggle kernel reported an error - see the log.")
        _update_run(project, run_id, status="error", stage="error", error=msg)
    shutil.rmtree(download, ignore_errors=True)


# ---------------------------------------------------------------- local pipeline

def _local_pipeline(project: str, run_id: str) -> None:
    run = load_run(project, run_id)
    cfg = run["config"]
    run_dir = _run_dir(project, run_id)
    ds = storage.require_project(project) / "dataset"
    annotation_service.write_data_yaml(project)

    model_source = run["model"]
    if model_source.endswith(".pt"):
        cached = storage.MODELS_DIR / model_source  # cache official weights once
        if cached.exists() and time.time() - cached.stat().st_mtime > 120 and not zipfile.is_zipfile(cached):
            cached.unlink()  # truncated by an interrupted download - fetch again
        model_source = str(cached)
    script = train_script.render({
        "run_id": run_id,
        "model": model_source,
        "pretrained": cfg["pretrained"],
        "names": run["classes"],
        "epochs": cfg["epochs"],
        "batch": cfg["batch"],
        "imgsz": cfg["imgsz"],
        "optimizer": cfg["optimizer"],
        "lr0": cfg["lr0"],
        "patience": cfg.get("patience", 100),
        "device": cfg.get("local_device", "auto"),
        "dataset_root": str(ds.resolve()),
        "work_dir": str(run_dir.resolve()),
        "project_dir": str((run_dir / "runs").resolve()),
        "artifact_dir": str((run_dir / "artifacts").resolve()),
        "workers": 2 if os.name == "nt" else None,
        "export_onnx": True,
    })
    (run_dir / "train.py").write_text(script, encoding="utf-8")

    _update_run(project, run_id, status="running", stage="running")
    storage.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    log_f = open(run_dir / "train.log", "a", encoding="utf-8")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen([sys.executable, "-u", str(run_dir / "train.py")], cwd=str(storage.MODELS_DIR),
                            stdout=log_f, stderr=subprocess.STDOUT, env=env, creationflags=creationflags)
    _procs[run_id] = proc
    _update_run(project, run_id, pid=proc.pid)
    try:
        while proc.poll() is None:
            _update_local_progress(project, run_id)
            time.sleep(3)
    finally:
        log_f.close()
        _procs.pop(run_id, None)

    if load_run(project, run_id).get("status") == "cancelled":
        return
    _collect_artifacts(project, run_id, run_dir / "runs")
    run = _finalize_metrics(project, run_id)
    if proc.returncode == 0 and "best.pt" in run.get("artifacts", []):
        _update_run(project, run_id, status="completed", stage="completed")
    else:
        _update_run(project, run_id, status="error", stage="error",
                    error=f"Training process exited with code {proc.returncode} - see the log.")


def _update_local_progress(project: str, run_id: str) -> None:
    csv_path = _run_dir(project, run_id) / "runs" / "train" / "results.csv"
    if csv_path.exists():
        rows = _read_results_csv(csv_path)["rows"]
        run = load_run(project, run_id)
        _update_run(project, run_id, progress={"epoch": len(rows), "epochs": run["config"]["epochs"]})


def cancel_run(project: str, run_id: str) -> dict:
    run = load_run(project, run_id)
    if run["status"] in TERMINAL:
        return run
    if run["backend"] == "local":
        proc = _procs.get(run_id)
        if proc and proc.poll() is None:
            if os.name == "nt":  # kill dataloader workers too
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.terminate()
        _append_log(project, run_id, "[yoloflow] run cancelled by user")
        return _update_run(project, run_id, status="cancelled", stage="cancelled")
    # The Kaggle API has no kernel-cancel endpoint: stop tracking and point the user to the UI.
    _append_log(project, run_id, "[yoloflow] stopped tracking - cancel the kernel on kaggle.com if it is still running")
    return _update_run(project, run_id, status="cancelled", stage="cancelled")


# ---------------------------------------------------------------- artifacts & metrics

def _collect_artifacts(project: str, run_id: str, search_root: Path) -> None:
    dst = _run_dir(project, run_id) / "artifacts"
    dst.mkdir(exist_ok=True)
    if not search_root.exists():
        return
    candidates = sorted(search_root.rglob("*"), key=lambda p: ("artifacts" not in p.parts, len(p.parts)))
    for p in candidates:
        if not p.is_file():
            continue
        keep = p.name in ARTIFACT_NAMES or (p.suffix == ".png" and "curve" in p.name.lower())
        if keep and not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)


def _read_results_csv(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return {"columns": [], "rows": []}
    columns = [c.strip() for c in rows[0]]
    data = []
    for r in rows[1:]:
        try:
            data.append([float(v) for v in r])
        except ValueError:
            continue
    return {"columns": columns, "rows": data}


def _finalize_metrics(project: str, run_id: str) -> dict:
    art = _run_dir(project, run_id) / "artifacts"
    metrics = {}
    mj = storage.read_json(art / "metrics.json", {})
    final = (mj or {}).get("metrics") or {}
    if (art / "results.csv").exists():
        res = _read_results_csv(art / "results.csv")
        if res["rows"]:
            last = dict(zip(res["columns"], res["rows"][-1]))
            final = {**last, **final}
    keymap = {
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
    }
    for k, src in keymap.items():
        if src in final:
            metrics[k] = round(float(final[src]), 4)
    if mj.get("elapsed_seconds"):
        metrics["elapsed_seconds"] = mj["elapsed_seconds"]
    artifacts = sorted(p.name for p in art.iterdir() if p.is_file())
    return _update_run(project, run_id, metrics=metrics or None, artifacts=artifacts)


def results_table(project: str, run_id: str) -> dict:
    run_dir = _run_dir(project, run_id)
    for p in (run_dir / "artifacts" / "results.csv", run_dir / "runs" / "train" / "results.csv"):
        if p.exists():
            return _read_results_csv(p)
    return {"columns": [], "rows": []}


def artifact_path(project: str, run_id: str, name: str) -> Path:
    art = _run_dir(project, run_id) / "artifacts"
    path = storage.ensure_within(art, art / name)
    if not path.is_file():
        raise NotFound("Artifact not found.")
    return path


def read_log(project: str, run_id: str, max_chars: int = 200_000) -> str:
    run_dir = _run_dir(project, run_id)
    parts = []
    for name in ("train.log", "remote.log"):
        p = run_dir / name
        if p.exists() and p.stat().st_size:
            with open(p, "rb") as f:
                size = p.stat().st_size
                f.seek(max(0, size - max_chars))
                text = f.read().decode("utf-8", errors="replace")
            if name == "remote.log":
                parts.append("\n===== Kaggle kernel log =====\n")
            parts.append(text)
    # Strip ANSI colors and collapse tqdm carriage-return redraws into their final state.
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", "".join(parts))
    lines = []
    for line in text.split("\n"):
        segs = [s for s in line.split("\r") if s.strip()]
        lines.append(segs[-1] if segs else "")
    return "\n".join(lines)[-max_chars:]


def refresh_run(project: str, run_id: str) -> dict:
    """Restart a dead Kaggle poller (e.g. after a crash) for a run that has a kernel."""
    run = load_run(project, run_id)
    if run["backend"] == "kaggle" and run["status"] in ACTIVE and run.get("kaggle", {}).get("kernel_ref"):
        t = _threads.get(run_id)
        if t is None or not t.is_alive():
            _spawn(project, run_id, _kaggle_poll)
    return load_run(project, run_id)


def resume_all() -> None:
    """Called on startup: resume Kaggle pollers, mark interrupted local runs."""
    for meta in storage.list_projects():
        project = meta["name"]
        try:
            runs = list_runs(project)
        except Exception:  # noqa: BLE001
            continue
        for run in runs:
            if run.get("status") not in ACTIVE:
                continue
            if run["backend"] == "kaggle" and run.get("kaggle", {}).get("kernel_ref"):
                _spawn(project, run["id"], _kaggle_poll)
            else:
                _update_run(project, run["id"], status="error", stage="error",
                            error="Interrupted by a server restart.")
