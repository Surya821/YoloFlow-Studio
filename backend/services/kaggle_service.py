"""Kaggle API wrapper: credentials, dataset packaging/upload, kernel generation/push,
status polling, log retrieval and artifact download."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path

from core import storage
from core.storage import BadRequest
from services import annotation_service, train_script

_env_lock = threading.Lock()


class KaggleError(Exception):
    pass


# ---------------------------------------------------------------- auth

def _credentials(username: str | None = None, key: str | None = None) -> tuple[str, str]:
    cfg = storage.load_config()["kaggle"]
    username = (username if username is not None else cfg.get("username", "")).strip()
    key = (key if key is not None else cfg.get("key", "")).strip()
    if not username or not key:
        raise BadRequest("Kaggle username and API key are not configured.")
    return username, key


def get_api(username: str | None = None, key: str | None = None):
    """Build an authenticated KaggleApi without touching ~/.kaggle/kaggle.json.

    `import kaggle` authenticates at import time (and calls exit() on failure), so the
    extended API class is imported directly and credentials are fed via environment.
    """
    username, key = _credentials(username, key)
    from kaggle.api.kaggle_api_extended import KaggleApi

    env = {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key}
    if key.startswith("KGAT_"):  # new-style API token
        env["KAGGLE_API_TOKEN"] = key
    with _env_lock:
        saved = {k: os.environ.get(k) for k in (*env, "KAGGLE_API_TOKEN")}
        os.environ.update(env)
        try:
            api = KaggleApi()
            api.authenticate()
        except SystemExit as exc:
            raise KaggleError("Kaggle authentication failed - check username and key.") from exc
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return api, username


def verify(username: str | None = None, key: str | None = None) -> dict:
    try:
        api, user = get_api(username, key)
        api.kernels_list(mine=True, page_size=1)  # real round-trip; legacy auth is lazy
        return {"ok": True, "username": user, "message": f"Authenticated as {user}"}
    except BadRequest:
        raise
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "401" in msg or "Unauthorized" in msg:
            msg = "Kaggle rejected the credentials (401 Unauthorized)."
        return {"ok": False, "username": username, "message": msg[:400]}


# ---------------------------------------------------------------- naming

def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:50].strip("-")


def dataset_slug(project: str) -> str:
    slug = slugify(f"{project}-yolo-dataset")
    return slug if len(slug) >= 6 else f"{slug}-data"


def kernel_slug(project: str) -> str:
    return slugify(f"{project}-yolo-train")


# ---------------------------------------------------------------- dataset

def package_dataset(project: str, staging: Path, run_id: str) -> Path:
    """Zip images/, labels/ and data.yaml into staging/dataset.zip with Kaggle metadata."""
    ds = storage.require_project(project) / "dataset"
    annotation_service.write_data_yaml(project)
    if annotation_service.split_counts(project)["train"] + annotation_service.split_counts(project)["val"] == 0:
        raise BadRequest("No annotated images - label some frames before training.")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    zip_path = staging / "dataset.zip"
    classes = storage.load_meta(project)["classes"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for sub in ("images", "labels"):
            for p in (ds / sub).rglob("*"):
                if p.is_file() and p.suffix != ".cache":  # skip Ultralytics label caches
                    zf.write(p, f"dataset/{p.relative_to(ds).as_posix()}")
        names = {i: c["name"] for i, c in enumerate(classes)}
        zf.writestr("dataset/data.yaml",
                    "path: .\ntrain: images/train\nval: images/val\n"
                    f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in names.items()))
        zf.writestr("dataset/manifest.json", json.dumps({"run_id": run_id, "project": project,
                                                          "created_at": time.time()}))
    return zip_path


def push_dataset(api, username: str, project: str, staging: Path, run_id: str) -> tuple[str, bool]:
    slug = dataset_slug(project)
    ref = f"{username}/{slug}"
    storage.write_json(staging / "dataset-metadata.json", {
        "title": slug,
        "id": ref,
        "licenses": [{"name": "CC0-1.0"}],
    })
    exists = True
    try:
        api.dataset_status(ref)
    except Exception:  # noqa: BLE001
        exists = False
    if exists:
        resp = api.dataset_create_version(str(staging), version_notes=f"YOLOFlow run {run_id}",
                                          quiet=True, convert_to_csv=False, dir_mode="zip")
    else:
        resp = api.dataset_create_new(str(staging), public=False, quiet=True,
                                      convert_to_csv=False, dir_mode="zip")
    err = getattr(resp, "error", None)
    if err:
        raise KaggleError(f"Dataset upload failed: {err}")
    return ref, not exists


def wait_dataset_ready(api, ref: str, timeout: float = 900, on_tick=None) -> None:
    deadline = time.time() + timeout
    time.sleep(20)  # give Kaggle time to register the new version before trusting "ready"
    while time.time() < deadline:
        try:
            status = str(api.dataset_status(ref) or "").lower()
        except Exception as exc:  # noqa: BLE001
            status = f"unknown ({exc})"
        if on_tick:
            on_tick(status)
        if "ready" in status:
            return
        if "error" in status:
            raise KaggleError(f"Kaggle failed to process the dataset: {status}")
        time.sleep(10)
    raise KaggleError("Timed out waiting for Kaggle to process the dataset.")


# ---------------------------------------------------------------- kernel

def build_kernel(username: str, project: str, run_id: str, dataset_ref: str, cfg: dict,
                 model_source: str, folder: Path) -> str:
    import ultralytics

    slug = kernel_slug(project)
    ref = f"{username}/{slug}"
    classes = [c["name"] for c in storage.load_meta(project)["classes"]]
    script = train_script.render({
        "run_id": run_id,
        "model": model_source,
        "pretrained": cfg["pretrained"],
        "names": classes,
        "epochs": cfg["epochs"],
        "batch": cfg["batch"],
        "imgsz": cfg["imgsz"],
        "optimizer": cfg["optimizer"],
        "lr0": cfg["lr0"],
        "patience": cfg.get("patience", 100),
        "device": cfg.get("kaggle_device", "0"),
        "search_root": "/kaggle/input",
        "work_dir": "/kaggle/working",
        "project_dir": "/kaggle/working/runs",
        "artifact_dir": "/kaggle/working/artifacts",
        "export_onnx": True,
        "pip_spec": f"ultralytics=={ultralytics.__version__}",
    })
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "train.py").write_text(script, encoding="utf-8")
    storage.write_json(folder / "kernel-metadata.json", {
        "id": ref,
        "title": slug,  # title must slugify to the id or Kaggle creates a different kernel
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    })
    return ref


def push_kernel(api, folder: Path, accelerator: str | None) -> dict:
    resp = api.kernels_push(str(folder), acc=accelerator or None)
    err = getattr(resp, "error", None)
    if err:
        raise KaggleError(f"Kernel push failed: {err}")
    return {"url": getattr(resp, "url", None), "version": getattr(resp, "version_number", None)}


def kernel_status(api, ref: str) -> tuple[str, str | None]:
    resp = api.kernels_status(ref)
    status = getattr(resp, "status", resp)
    name = getattr(status, "name", None) or str(status).split(".")[-1]
    return name.upper(), getattr(resp, "failure_message", None) or None


def _parse_kaggle_log(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        entries = json.loads(raw)
        if isinstance(entries, list):
            return "".join(str(e.get("data", "")) for e in entries if isinstance(e, dict))
    except json.JSONDecodeError:
        # Logs fetched mid-run can be a truncated JSON array; salvage the "data" fields.
        chunks = re.findall(r'"data"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if chunks:
            try:
                return "".join(json.loads(f'"{c}"') for c in chunks)
            except json.JSONDecodeError:
                pass
    return raw


def kernel_log(api, ref: str) -> str:
    try:
        return _parse_kaggle_log(api.kernels_logs(ref))
    except Exception:  # noqa: BLE001 - logs are best effort while running
        return ""


def download_output(api, ref: str, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    token = None
    for _ in range(50):
        got, token = api.kernels_output(ref, str(dest), force=True, quiet=True, page_token=token, page_size=50)
        files.extend(got or [])
        if not token:
            break
    return files


def kaggle_log_from_download(dest: Path) -> str:
    for p in dest.rglob("*.log"):
        try:
            return _parse_kaggle_log(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return ""
