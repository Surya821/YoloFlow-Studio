"""File-system storage layer: workspace layout, JSON metadata and path safety."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE = Path(os.environ.get("YOLOFLOW_WORKSPACE", BASE_DIR / "workspace")).resolve()
PROJECTS_DIR = WORKSPACE / "projects"
MODELS_DIR = WORKSPACE / "models"
CONFIG_PATH = WORKSPACE / "config.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
IMAGE_FOLDERS = ("raw_media", "extracted_frames")

DEFAULT_TRAINING = {
    "family": "11",
    "variant": "n",
    "pretrained": True,
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "optimizer": "auto",
    "lr0": 0.01,
    "patience": 100,
    "accelerator": "NvidiaTeslaT4",
    "kaggle_device": "0",
    "local_device": "auto",
}

DEFAULT_CONFIG = {
    "kaggle": {"username": "", "key": "", "verified": False, "verified_at": None},
    "training_defaults": DEFAULT_TRAINING,
}


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


# ---------------------------------------------------------------- locking

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def lock_for(key: str) -> threading.RLock:
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.RLock()
        return _locks[key]


# ---------------------------------------------------------------- json io

def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return copy.deepcopy(default)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    # os.replace can transiently fail on Windows while another handle reads the file.
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


# ---------------------------------------------------------------- names & paths

_PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def sanitize_project_name(name: str) -> str:
    cleaned = re.sub(r"\s+", "-", (name or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", cleaned)[:64].strip("-_")
    if not cleaned or not _PROJECT_RE.match(cleaned):
        raise BadRequest("Project name must contain letters or digits (A-Z, 0-9, '-', '_').")
    return cleaned


def safe_filename(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "file"
    ext = re.sub(r"[^A-Za-z0-9.]", "", ext).lower()
    return f"{stem[:100]}{ext}"


def unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    stem, ext = os.path.splitext(filename)
    i = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{i}{ext}"
        i += 1
    return candidate


def ensure_within(parent: Path, child: Path) -> Path:
    resolved = child.resolve()
    if parent.resolve() not in resolved.parents and resolved != parent.resolve():
        raise BadRequest("Invalid path.")
    return resolved


def project_dir(name: str) -> Path:
    if not _PROJECT_RE.match(name or ""):
        raise BadRequest("Invalid project name.")
    return PROJECTS_DIR / name


def require_project(name: str) -> Path:
    path = project_dir(name)
    if not (path / "project_meta.json").exists():
        raise NotFound(f"Project '{name}' not found.")
    return path


def project_file(project: str, folder: str, filename: str) -> Path:
    if folder not in IMAGE_FOLDERS:
        raise BadRequest(f"Unknown folder '{folder}'.")
    root = require_project(project) / folder
    if safe_filename(filename) != filename:
        raise BadRequest("Invalid file name.")
    path = ensure_within(root, root / filename)
    if not path.exists():
        raise NotFound(f"File '{folder}/{filename}' not found.")
    return path


PROJECT_SUBDIRS = (
    "raw_media",
    "extracted_frames",
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val",
    "outputs",
    ".cache/thumbs",
)


def init_workspace() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        write_json(CONFIG_PATH, DEFAULT_CONFIG)


def create_project(name: str) -> dict:
    name = sanitize_project_name(name)
    path = project_dir(name)
    with lock_for(f"project:{name}"):
        if path.exists():
            raise BadRequest(f"Project '{name}' already exists.")
        for sub in PROJECT_SUBDIRS:
            (path / sub).mkdir(parents=True, exist_ok=True)
        meta = {
            "name": name,
            "created_at": time.time(),
            "classes": [],
            "media": {},
        }
        write_json(path / "project_meta.json", meta)
    return meta


def delete_project(name: str) -> None:
    path = require_project(name)
    with lock_for(f"project:{name}"):
        shutil.rmtree(path, ignore_errors=True)


def list_projects() -> list[dict]:
    out = []
    if not PROJECTS_DIR.exists():
        return out
    for d in sorted(PROJECTS_DIR.iterdir()):
        meta = read_json(d / "project_meta.json")
        if meta:
            out.append(meta)
    return out


def load_meta(name: str) -> dict:
    path = require_project(name)
    meta = read_json(path / "project_meta.json", {})
    meta.setdefault("classes", [])
    meta.setdefault("media", {})
    return meta


def update_meta(name: str, fn: Callable[[dict], Any]) -> Any:
    """Atomically read-modify-write project_meta.json. Returns fn's result."""
    path = require_project(name)
    with lock_for(f"project:{name}"):
        meta = load_meta(name)
        result = fn(meta)
        write_json(path / "project_meta.json", meta)
        return result


# ---------------------------------------------------------------- global config

def load_config() -> dict:
    cfg = read_json(CONFIG_PATH, DEFAULT_CONFIG)
    merged = copy.deepcopy(DEFAULT_CONFIG)
    merged["kaggle"].update(cfg.get("kaggle", {}))
    merged["training_defaults"].update(cfg.get("training_defaults", {}))
    return merged


def update_config(fn: Callable[[dict], Any]) -> dict:
    with lock_for("config"):
        cfg = load_config()
        fn(cfg)
        write_json(CONFIG_PATH, cfg)
        return cfg
