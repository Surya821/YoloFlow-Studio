"""Run trained / uploaded / COCO YOLO models on images, video files and live RTSP streams."""
from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core import storage
from core.jobs import Job
from core.storage import BadRequest, NotFound
from services import media_service, model_registry


def _inference_dir(project: str) -> Path:
    d = storage.require_project(project) / "inference"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_detections(result, keep: set[str] | None) -> list[dict]:
    out = []
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return out
    names = result.names
    h, w = result.orig_shape
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c, p in zip(xyxy, clss, confs):
        name = names[int(c)]
        if keep and name not in keep:
            continue
        out.append({"cls": int(c), "class_name": name, "conf": round(float(p), 4),
                    "x1": float(x1) / w, "y1": float(y1) / h, "x2": float(x2) / w, "y2": float(y2) / h})
    return out


def _class_ids(model, classes: list[str] | None) -> list[int] | None:
    if not classes:
        return None
    names = model_registry.class_names(model)
    ids = [i for i, n in enumerate(names) if n in set(classes)]
    return ids or None


def _counts(dets: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in dets:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- images

def detect_image(project: str, model_key: str, conf: float, iou: float, imgsz: int | None,
                 classes: list[str] | None, data: bytes | None = None, image_id: str | None = None) -> dict:
    if data is not None:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise BadRequest("Could not decode the image.")
        source = frame
    elif image_id:
        folder, _, filename = image_id.partition("/")
        source = str(storage.project_file(project, folder, filename))
    else:
        raise BadRequest("Provide an image file or a project image id.")

    model, lock = model_registry.get_cached(model_registry.resolve(project, model_key))
    kwargs = {"conf": conf, "iou": iou, "verbose": False, "classes": _class_ids(model, classes)}
    if imgsz:
        kwargs["imgsz"] = imgsz
    with lock:
        t0 = time.perf_counter()
        result = model.predict(source, **kwargs)[0]
        elapsed = (time.perf_counter() - t0) * 1000
    dets = _parse_detections(result, None)
    h, w = result.orig_shape
    return {"width": w, "height": h, "detections": dets, "counts": _counts(dets),
            "inference_ms": round(elapsed, 1), "model": model_key}


# ---------------------------------------------------------------- videos

async def save_video_upload(project: str, upload) -> Path:
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in storage.VIDEO_EXTS:
        raise BadRequest("Unsupported video type.")
    d = _inference_dir(project) / "_uploads"
    d.mkdir(exist_ok=True)
    dest = storage.unique_path(d, storage.safe_filename(upload.filename))
    with open(dest, "wb") as f:
        while chunk := await upload.read(4 * 1024 * 1024):
            f.write(chunk)
    return dest


def resolve_video_source(project: str, media_filename: str | None, upload_path: Path | None) -> Path:
    if upload_path:
        return upload_path
    if media_filename:
        path = storage.require_project(project) / "raw_media" / storage.safe_filename(media_filename)
        if not path.exists() or media_service.media_kind(path.name) != "video":
            raise NotFound("Video not found in project media.")
        return path
    raise BadRequest("Provide a video file or choose a project video.")


def run_video(job: Job, project: str, src: Path, model_key: str, conf: float, iou: float,
              imgsz: int | None, classes: list[str] | None, delete_source: bool) -> dict:
    model = model_registry.load_fresh(model_registry.resolve(project, model_key))  # private instance
    class_ids = _class_ids(model, classes)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if 0 < fps < 240 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2  # H.264 needs even dimensions
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    out_dir = _inference_dir(project) / run_id
    out_dir.mkdir(parents=True)
    writer, out_path, playable = media_service.open_playable_writer(out_dir / "annotated", fps, (w, h))

    totals: dict[str, int] = {}
    peak: dict[str, int] = {}
    frames = frames_with = 0
    t0 = time.time()
    kwargs = {"conf": conf, "iou": iou, "verbose": False, "classes": class_ids}
    if imgsz:
        kwargs["imgsz"] = imgsz
    job.update(0, total, "Processing video")
    try:
        while True:
            job.check_cancelled()
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, **kwargs)[0]
            plotted = result.plot(line_width=max(1, round(w / 640) + 1))
            if plotted.shape[1] != w or plotted.shape[0] != h:
                plotted = cv2.resize(plotted, (w, h))
            writer.write(plotted)
            dets = _parse_detections(result, None)
            frames += 1
            if dets:
                frames_with += 1
            for name, n in _counts(dets).items():
                totals[name] = totals.get(name, 0) + n
                peak[name] = max(peak.get(name, 0), n)
            if frames % 5 == 0 or frames == 1:
                rate = frames / max(1e-6, time.time() - t0)
                job.update(frames, max(total, frames), f"{frames} frames · {rate:.1f} fps")
    finally:
        cap.release()
        writer.release()
        if delete_source:
            src.unlink(missing_ok=True)

    summary = {
        "id": run_id,
        "source": src.name,
        "model": model_key,
        "conf": conf,
        "output": out_path.name,
        "browser_playable": playable,
        "frames": frames,
        "frames_with_detections": frames_with,
        "fps": round(fps, 2),
        "processing_seconds": round(time.time() - t0, 1),
        "detections_total": totals,
        "peak_per_frame": peak,
        "created_at": time.time(),
    }
    storage.write_json(out_dir / "summary.json", summary)
    job.update(frames, frames, f"Done - {frames} frames")
    return summary


def list_video_results(project: str) -> list[dict]:
    out = []
    for d in _inference_dir(project).iterdir():
        s = storage.read_json(d / "summary.json")
        if s:
            out.append(s)
    return sorted(out, key=lambda s: s.get("created_at", 0), reverse=True)


def video_result_file(project: str, result_id: str) -> Path:
    root = _inference_dir(project)
    d = storage.ensure_within(root, root / result_id)
    s = storage.read_json(d / "summary.json")
    if not s:
        raise NotFound("Result not found.")
    return d / s["output"]


def delete_video_result(project: str, result_id: str) -> None:
    root = _inference_dir(project)
    d = storage.ensure_within(root, root / result_id)
    if not (d / "summary.json").exists():
        raise NotFound("Result not found.")
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- live RTSP

class LiveInference:
    """Reads a stream (via StreamSession) and runs the model on the newest frame in a worker thread."""

    def __init__(self, project: str, url: str, model_key: str, conf: float, iou: float,
                 imgsz: int | None, classes: list[str] | None):
        self.id = uuid.uuid4().hex[:10]
        self.project = project
        self.model_key = model_key
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes
        self.model = model_registry.load_fresh(model_registry.resolve(project, model_key))
        self.stream = media_service.StreamSession(project, url, "live_infer")
        self.error: str | None = None
        self.infer_fps = 0.0
        self.counts: dict[str, int] = {}
        self.detections: list[dict] = []
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"live-infer-{self.id}")
        self._thread.start()

    def _run(self):
        last_seq = -1
        while not self._stop.is_set():
            seq, frame = self.stream.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq
            try:
                kwargs = {"conf": self.conf, "iou": self.iou, "verbose": False,
                          "classes": _class_ids(self.model, self.classes)}
                if self.imgsz:
                    kwargs["imgsz"] = self.imgsz
                t0 = time.perf_counter()
                result = self.model.predict(frame, **kwargs)[0]
                dt = time.perf_counter() - t0
                plotted = result.plot(line_width=max(1, round(frame.shape[1] / 640) + 1))
                dets = _parse_detections(result, None)
                inst = 1.0 / max(1e-3, dt)
                self.infer_fps = inst if not self.infer_fps else 0.85 * self.infer_fps + 0.15 * inst
                with self._lock:
                    self._frame = plotted
                    self._seq += 1
                    self.detections = dets
                    self.counts = _counts(dets)
                self.error = None
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)[:300]
                time.sleep(0.5)

    def latest(self):
        with self._lock:
            return self._seq, self._frame

    @property
    def alive(self) -> bool:
        return not self._stop.is_set()

    def update(self, conf: float | None, iou: float | None, classes: list[str] | None):
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        if classes is not None:
            self.classes = classes or None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self.stream.stop()

    def to_dict(self) -> dict:
        s = self.stream.to_dict()
        return {
            "id": self.id,
            "project": self.project,
            "url": self.stream.url,
            "model": self.model_key,
            "conf": self.conf,
            "iou": self.iou,
            "classes": self.classes or [],
            "status": s["status"],
            "stream_error": s["error"],
            "error": self.error,
            "width": s["width"],
            "height": s["height"],
            "stream_fps": s["fps"],
            "infer_fps": round(self.infer_fps, 1),
            "counts": self.counts,
            "detections": len(self.detections),
        }


class LiveManager:
    def __init__(self):
        self._sessions: dict[str, LiveInference] = {}
        self._lock = threading.Lock()

    def start(self, project: str, url: str, model_key: str, conf: float, iou: float,
              imgsz: int | None, classes: list[str] | None) -> LiveInference:
        storage.require_project(project)
        if not url.strip():
            raise BadRequest("Stream URL is required.")
        session = LiveInference(project, url, model_key, conf, iou, imgsz, classes)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, sid: str) -> LiveInference:
        s = self._sessions.get(sid)
        if s is None:
            raise NotFound("Live session not found.")
        return s

    def for_project(self, project: str) -> list[LiveInference]:
        return [s for s in self._sessions.values() if s.project == project]

    def stop(self, sid: str) -> None:
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s is None:
            raise NotFound("Live session not found.")
        s.stop()

    def stop_project(self, project: str) -> None:
        for s in self.for_project(project):
            self.stop(s.id)

    def stop_all(self) -> None:
        for sid in list(self._sessions):
            try:
                self.stop(sid)
            except Exception:  # noqa: BLE001
                pass


live = LiveManager()
