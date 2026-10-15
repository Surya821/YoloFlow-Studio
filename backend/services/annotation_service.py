"""Annotation storage (YOLO .txt labels), train/val splitting, data.yaml generation,
class management and AI-assisted pre-labeling with Ultralytics YOLO."""
from __future__ import annotations

import os
import shutil
import zlib
from pathlib import Path

import yaml

from core import storage
from core.catalog import CLASS_COLORS, COCO_CLASSES
from core.storage import BadRequest, NotFound
from services import model_registry

VAL_RATIO = 0.2
SPLITS = ("train", "val")
_FOLDER_PREFIX = {"raw_media": "raw", "extracted_frames": "frm"}


# ---------------------------------------------------------------- ids & paths

def parse_image_id(image_id: str) -> tuple[str, str]:
    folder, _, filename = (image_id or "").partition("/")
    if folder not in storage.IMAGE_FOLDERS or not filename:
        raise BadRequest(f"Invalid image id '{image_id}'.")
    return folder, filename


def dataset_stem(folder: str, filename: str) -> str:
    return f"{_FOLDER_PREFIX[folder]}__{os.path.splitext(filename)[0]}"


def split_for(stem: str) -> str:
    """Deterministic ~80/20 split keyed on the file name, stable across re-saves."""
    return "val" if zlib.crc32(stem.encode("utf-8")) % 100 < VAL_RATIO * 100 else "train"


def _dataset_dir(project: str) -> Path:
    return storage.require_project(project) / "dataset"


def _label_path(project: str, stem: str) -> tuple[Path | None, str | None]:
    ds = _dataset_dir(project)
    for split in SPLITS:
        p = ds / "labels" / split / f"{stem}.txt"
        if p.exists():
            return p, split
    return None, None


# ---------------------------------------------------------------- listing

def list_images(project: str) -> list[dict]:
    root = storage.require_project(project)
    ds = root / "dataset"
    labeled: dict[str, tuple[str, int]] = {}
    for split in SPLITS:
        for p in (ds / "labels" / split).glob("*.txt"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    n = sum(1 for line in f if line.strip())
            except OSError:
                n = 0
            labeled[p.stem] = (split, n)

    images = []
    for folder in storage.IMAGE_FOLDERS:
        d = root / folder
        if not d.exists():
            continue
        for p in sorted(d.iterdir(), key=lambda x: x.name):
            if p.suffix.lower() not in storage.IMAGE_EXTS:
                continue
            stem = dataset_stem(folder, p.name)
            split, count = labeled.get(stem, (None, 0))
            images.append({
                "id": f"{folder}/{p.name}",
                "folder": folder,
                "filename": p.name,
                "labeled": split is not None,
                "box_count": count,
                "split": split,
            })
    return images


# ---------------------------------------------------------------- read / write labels

def _read_boxes(path: Path) -> list[dict]:
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def get_annotation(project: str, image_id: str) -> dict:
    folder, filename = parse_image_id(image_id)
    storage.project_file(project, folder, filename)
    path, split = _label_path(project, dataset_stem(folder, filename))
    return {"image_id": image_id, "labeled": path is not None, "split": split,
            "boxes": _read_boxes(path) if path else []}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def save_annotation(project: str, image_id: str, boxes: list[dict]) -> dict:
    folder, filename = parse_image_id(image_id)
    src = storage.project_file(project, folder, filename)
    n_classes = len(storage.load_meta(project)["classes"])
    lines = []
    for b in boxes:
        cls = int(b["cls"])
        if not 0 <= cls < n_classes:
            raise BadRequest(f"Class id {cls} does not exist.")
        # Clamp to the image and drop degenerate boxes.
        x1 = _clamp01(b["cx"] - b["w"] / 2)
        y1 = _clamp01(b["cy"] - b["h"] / 2)
        x2 = _clamp01(b["cx"] + b["w"] / 2)
        y2 = _clamp01(b["cy"] + b["h"] / 2)
        w, h = x2 - x1, y2 - y1
        if w <= 1e-4 or h <= 1e-4:
            continue
        lines.append(f"{cls} {x1 + w / 2:.6f} {y1 + h / 2:.6f} {w:.6f} {h:.6f}")

    stem = dataset_stem(folder, filename)
    split = split_for(stem)
    ds = _dataset_dir(project)
    with storage.lock_for(f"labels:{project}"):
        _remove_from_dataset(ds, stem, src.suffix)
        label = ds / "labels" / split / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        img_dst = ds / "images" / split / f"{stem}{src.suffix.lower()}"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, img_dst)  # hard link saves disk space on the same volume
        except OSError:
            shutil.copy2(src, img_dst)
    write_data_yaml(project)
    return {"image_id": image_id, "labeled": True, "split": split, "box_count": len(lines)}


def _remove_from_dataset(ds: Path, stem: str, ext: str | None = None) -> None:
    for split in SPLITS:
        (ds / "labels" / split / f"{stem}.txt").unlink(missing_ok=True)
        img_dir = ds / "images" / split
        if ext:
            (img_dir / f"{stem}{ext.lower()}").unlink(missing_ok=True)
        for p in img_dir.glob(f"{stem}.*"):
            if p.stem == stem:
                p.unlink(missing_ok=True)


def delete_annotation(project: str, image_id: str) -> None:
    folder, filename = parse_image_id(image_id)
    with storage.lock_for(f"labels:{project}"):
        _remove_from_dataset(_dataset_dir(project), dataset_stem(folder, filename))


def delete_image(project: str, folder: str, filename: str) -> None:
    src = storage.project_file(project, folder, filename)
    delete_annotation(project, f"{folder}/{filename}")
    src.unlink(missing_ok=True)
    thumb = storage.require_project(project) / ".cache" / "thumbs" / f"{folder}__{filename}.jpg"
    thumb.unlink(missing_ok=True)


# ---------------------------------------------------------------- data.yaml & stats

def split_counts(project: str) -> dict:
    ds = _dataset_dir(project)
    return {s: sum(1 for _ in (ds / "labels" / s).glob("*.txt")) for s in SPLITS}


def write_data_yaml(project: str) -> Path:
    ds = _dataset_dir(project)
    classes = storage.load_meta(project)["classes"]
    counts = split_counts(project)
    data = {
        "path": str(ds.resolve()).replace("\\", "/"),
        "train": "images/train",
        # Ultralytics needs a non-empty val set; fall back to train for tiny datasets.
        "val": "images/val" if counts["val"] > 0 else "images/train",
        "nc": len(classes),
        "names": {i: c["name"] for i, c in enumerate(classes)},
    }
    out = ds / "data.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out


def dataset_stats(project: str) -> dict:
    ds = _dataset_dir(project)
    classes = storage.load_meta(project)["classes"]
    per_class = [0] * len(classes)
    images = {s: 0 for s in SPLITS}
    empty = 0
    for split in SPLITS:
        for p in (ds / "labels" / split).glob("*.txt"):
            images[split] += 1
            boxes = _read_boxes(p)
            if not boxes:
                empty += 1
            for b in boxes:
                if 0 <= b["cls"] < len(per_class):
                    per_class[b["cls"]] += 1
    return {
        "images": images,
        "labeled_total": images["train"] + images["val"],
        "background_images": empty,
        "boxes_total": sum(per_class),
        "per_class": [{"id": i, "name": c["name"], "color": c["color"], "boxes": per_class[i]}
                      for i, c in enumerate(classes)],
    }


# ---------------------------------------------------------------- classes

def _next_color(classes: list[dict]) -> str:
    used = {c["color"].upper() for c in classes}
    for color in CLASS_COLORS:
        if color.upper() not in used:
            return color
    return CLASS_COLORS[len(classes) % len(CLASS_COLORS)]


def _norm(name: str) -> str:
    return " ".join((name or "").strip().split())


def add_classes(project: str, names: list[str], color: str | None = None) -> list[dict]:
    def fn(meta):
        existing = {c["name"].lower() for c in meta["classes"]}
        for raw in names:
            name = _norm(raw)
            if not name or name.lower() in existing:
                continue
            meta["classes"].append({"name": name, "color": color or _next_color(meta["classes"])})
            existing.add(name.lower())
        return meta["classes"]

    classes = storage.update_meta(project, fn)
    write_data_yaml(project)
    return classes


def update_class(project: str, class_id: int, name: str | None, color: str | None) -> list[dict]:
    def fn(meta):
        if not 0 <= class_id < len(meta["classes"]):
            raise NotFound("Class not found.")
        if name is not None:
            new = _norm(name)
            if not new:
                raise BadRequest("Class name cannot be empty.")
            if any(i != class_id and c["name"].lower() == new.lower() for i, c in enumerate(meta["classes"])):
                raise BadRequest(f"Class '{new}' already exists.")
            meta["classes"][class_id]["name"] = new
        if color is not None:
            meta["classes"][class_id]["color"] = color
        return meta["classes"]

    classes = storage.update_meta(project, fn)
    write_data_yaml(project)
    return classes


def delete_class(project: str, class_id: int) -> list[dict]:
    """Delete a class and re-index every label file (drop its boxes, shift higher ids down)."""
    ds = _dataset_dir(project)
    with storage.lock_for(f"labels:{project}"):
        def fn(meta):
            if not 0 <= class_id < len(meta["classes"]):
                raise NotFound("Class not found.")
            meta["classes"].pop(class_id)
            return meta["classes"]

        classes = storage.update_meta(project, fn)
        for split in SPLITS:
            for p in (ds / "labels" / split).glob("*.txt"):
                boxes = _read_boxes(p)
                kept = []
                for b in boxes:
                    if b["cls"] == class_id:
                        continue
                    cls = b["cls"] - 1 if b["cls"] > class_id else b["cls"]
                    kept.append(f"{cls} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}")
                p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    write_data_yaml(project)
    return classes


# ---------------------------------------------------------------- AI auto-annotation

def list_autolabel_models(project: str) -> list[dict]:
    return model_registry.list_models(project)


def model_classes(project: str, model_key: str) -> list[str]:
    if model_key.startswith("base:"):
        model_registry.resolve(project, model_key)
        return COCO_CLASSES
    model, _ = model_registry.get_cached(model_registry.resolve(project, model_key))
    return model_registry.class_names(model)


def autolabel(project: str, image_id: str, model_key: str, conf: float = 0.25,
              class_filter: list[str] | None = None, iou: float = 0.5) -> dict:
    folder, filename = parse_image_id(image_id)
    src = storage.project_file(project, folder, filename)
    model, lock = model_registry.get_cached(model_registry.resolve(project, model_key))
    with lock:
        result = model.predict(str(src), conf=float(conf), iou=float(iou), verbose=False)[0]
    names = result.names
    wanted = {n.lower() for n in class_filter} if class_filter else None

    detections = []
    if result.boxes is not None and len(result.boxes):
        xywhn = result.boxes.xywhn.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for (cx, cy, w, h), c, p in zip(xywhn, clss, confs):
            name = names[int(c)]
            if wanted is not None and name.lower() not in wanted:
                continue
            detections.append({"class_name": name, "conf": float(p),
                               "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)})

    # Make sure every detected class exists in the project so boxes can be kept.
    classes = storage.load_meta(project)["classes"]
    missing = {d["class_name"] for d in detections} - {c["name"] for c in classes}
    if missing:
        lower = {c["name"].lower(): c["name"] for c in classes}
        truly_missing = [m for m in missing if m.lower() not in lower]
        if truly_missing:
            classes = add_classes(project, sorted(truly_missing))
    index = {c["name"].lower(): i for i, c in enumerate(classes)}
    boxes = [{**d, "cls": index[d["class_name"].lower()]} for d in detections]
    return {"boxes": boxes, "classes": classes}
