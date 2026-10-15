"""Media ingestion: uploads, video metadata, thumbnails, frame extraction and RTSP streaming/recording."""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

# Prefer TCP for RTSP: far fewer corrupted frames than UDP on typical networks.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402

from core import storage  # noqa: E402
from core.jobs import Job  # noqa: E402
from core.storage import BadRequest, NotFound  # noqa: E402
from services import annotation_service  # noqa: E402

CHUNK = 4 * 1024 * 1024


# ---------------------------------------------------------------- helpers

def _write_image(path: Path, frame, fmt: str, quality: int = 95) -> None:
    # imencode + raw write avoids cv2.imwrite's non-ASCII path issues on Windows.
    if fmt == "png":
        ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode frame.")
    path.write_bytes(buf.tobytes())


def _read_image(path: Path):
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def open_playable_writer(stem_path: Path, fps: float, size: tuple[int, int]):
    """Open a VideoWriter browsers can play: H.264 MP4 (Media Foundation on Windows / FFmpeg elsewhere),
    then VP8 WebM, then mp4v MP4 (download-only). Returns (writer, path, browser_playable)."""
    w, h = size
    candidates = []
    if os.name == "nt":
        candidates.append((cv2.CAP_MSMF, "avc1", ".mp4", True))
    candidates += [
        (cv2.CAP_FFMPEG, "avc1", ".mp4", True),
        (cv2.CAP_FFMPEG, "VP80", ".webm", True),
        (cv2.CAP_FFMPEG, "mp4v", ".mp4", False),
    ]
    for api_pref, fourcc, ext, playable in candidates:
        path = stem_path.with_suffix(ext)
        try:
            writer = cv2.VideoWriter(str(path), api_pref, cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
        except cv2.error:
            continue
        if writer.isOpened():
            return writer, path, playable
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError("No usable video encoder found in this OpenCV build.")


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {"readable": False}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "readable": True,
            "fps": round(fps, 3) if 0 < fps < 1000 else None,
            "frame_count": frames if frames > 0 else None,
            "duration": round(frames / fps, 2) if fps and 0 < fps < 1000 and frames > 0 else None,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def media_kind(filename: str) -> str | None:
    ext = os.path.splitext(filename)[1].lower()
    if ext in storage.IMAGE_EXTS:
        return "image"
    if ext in storage.VIDEO_EXTS:
        return "video"
    return None


def _register_media(project: str, filename: str, source: str, extra: dict | None = None) -> dict:
    path = storage.require_project(project) / "raw_media" / filename
    kind = media_kind(filename)
    entry = {"filename": filename, "type": kind, "source": source, "created_at": time.time(),
             "size": path.stat().st_size}
    if kind == "video":
        entry["info"] = video_info(path)
    if extra:
        entry.update(extra)

    def fn(meta):
        meta["media"][filename] = entry

    storage.update_meta(project, fn)
    return entry


# ---------------------------------------------------------------- uploads & listing

async def save_upload(project: str, upload) -> dict:
    root = storage.require_project(project) / "raw_media"
    filename = storage.safe_filename(upload.filename or "upload")
    if media_kind(filename) is None:
        raise BadRequest(f"Unsupported file type: {upload.filename}")
    dest = storage.unique_path(root, filename)
    with open(dest, "wb") as f:
        while chunk := await upload.read(CHUNK):
            f.write(chunk)
    return await asyncio.to_thread(_register_media, project, dest.name, "upload")


def list_media(project: str) -> list[dict]:
    root = storage.require_project(project) / "raw_media"
    meta = storage.load_meta(project)["media"]
    out = []
    for p in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        kind = media_kind(p.name)
        if not kind or not p.is_file():
            continue
        entry = meta.get(p.name)
        if entry is None:  # file dropped into the folder manually
            entry = _register_media(project, p.name, "filesystem")
        out.append(entry)
    return out


def delete_media(project: str, filename: str) -> None:
    kind = media_kind(filename)
    if kind == "image":
        annotation_service.delete_image(project, "raw_media", filename)
    else:
        path = storage.require_project(project) / "raw_media" / storage.safe_filename(filename)
        if not path.exists():
            raise NotFound("Media not found.")
        path.unlink()
        thumb = storage.require_project(project) / ".cache" / "thumbs" / f"raw_media__{filename}.jpg"
        thumb.unlink(missing_ok=True)

    def fn(meta):
        meta["media"].pop(filename, None)

    storage.update_meta(project, fn)


def thumbnail(project: str, folder: str, filename: str, width: int = 320) -> Path:
    src = storage.project_file(project, folder, filename)
    thumbs = storage.require_project(project) / ".cache" / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    dst = thumbs / f"{folder}__{filename}.jpg"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    if media_kind(filename) == "video":
        cap = cv2.VideoCapture(str(src))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 10)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise NotFound("Could not read video frame.")
    else:
        frame = _read_image(src)
        if frame is None:
            raise NotFound("Could not decode image.")
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    _write_image(dst, frame, "jpg", 80)
    return dst


# ---------------------------------------------------------------- frame extraction

def extract_frames(job: Job, project: str, filename: str, mode: str, fps: float, interval: int,
                   fmt: str, quality: int = 95, start_sec: float | None = None,
                   end_sec: float | None = None) -> dict:
    root = storage.require_project(project)
    src = root / "raw_media" / storage.safe_filename(filename)
    if not src.exists() or media_kind(src.name) != "video":
        raise NotFound("Video not found.")
    fmt = "png" if fmt == "png" else "jpg"
    out_dir = root / "extracted_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if not 0 < src_fps < 1000:
            src_fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        first = int((start_sec or 0) * src_fps)
        last = int(end_sec * src_fps) if end_sec else (total if total > 0 else None)
        if first > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)

        if mode == "interval":
            step = max(1, int(interval))
        else:
            target = float(fps)
            if target <= 0:
                raise BadRequest("FPS must be positive.")
            step = max(1.0, src_fps / target)

        span = (last - first) if last else None
        expected = int(span / step) + 1 if span else 0
        job.update(0, expected, f"Extracting from {src.name} at every {step:.2f} frames")

        stem = os.path.splitext(src.name)[0]
        idx = first
        next_pick = float(first)
        saved = 0
        while True:
            job.check_cancelled()
            if last is not None and idx >= last:
                break
            if idx >= round(next_pick):
                ok, frame = cap.read()
                if not ok:
                    break
                _write_image(out_dir / f"{stem}_{idx:07d}.{fmt}", frame, fmt, quality)
                saved += 1
                next_pick += step
                if saved % 5 == 0 or saved == 1:
                    job.update(saved, max(expected, saved), f"Extracted {saved} frames")
            else:
                if not cap.grab():  # skip without decoding
                    break
            idx += 1
        job.update(saved, saved, f"Done - {saved} frames extracted")
        return {"extracted": saved, "source": src.name, "format": fmt}
    finally:
        cap.release()


# ---------------------------------------------------------------- RTSP streaming

class StreamSession:
    def __init__(self, project: str, url: str, name: str):
        self.id = uuid.uuid4().hex[:10]
        self.project = project
        self.url = url.strip()
        self.name = storage.safe_filename(name or "stream").replace(".", "_") or "stream"
        self.status = "connecting"
        self.error: str | None = None
        self.width = 0
        self.height = 0
        self.fps_measured = 0.0
        self.fps_reported = 0.0
        self.started_at = time.time()

        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None

        self._writer = None
        self.recording_file: str | None = None
        self.recording_started: float | None = None
        self.frames_written = 0

        self._thread = threading.Thread(target=self._run, daemon=True, name=f"rtsp-{self.id}")
        self._thread.start()

    # -- reader loop
    def _open(self):
        src = int(self.url) if self.url.isdigit() else self.url  # "0" = local webcam
        cap = cv2.VideoCapture(src) if isinstance(src, int) else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        self.fps_reported = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return cap

    def _run(self):
        failures = 0
        last_t = None
        while not self._stop.is_set():
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    self.status = "reconnecting" if self._seq else "connecting"
                    self.error = "Unable to open stream - retrying"
                    self._stop.wait(2.0)
                    continue
                failures = 0
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 25:
                    self._cap.release()
                    self._cap = None
                    self.status = "reconnecting"
                    self.error = "Stream dropped - reconnecting"
                else:
                    time.sleep(0.02)
                continue
            failures = 0
            now = time.time()
            if last_t is not None:
                inst = 1.0 / max(1e-3, now - last_t)
                self.fps_measured = inst if not self.fps_measured else 0.9 * self.fps_measured + 0.1 * inst
            last_t = now
            self.height, self.width = frame.shape[:2]
            self.status = "live"
            self.error = None
            with self._lock:
                self._frame = frame
                self._seq += 1
                if self._writer is not None:
                    if (frame.shape[1], frame.shape[0]) != self._writer_size:
                        frame = cv2.resize(frame, self._writer_size)
                    self._writer.write(frame)
                    self.frames_written += 1
        if self._cap is not None:
            self._cap.release()
        self._close_writer()
        self.status = "stopped"

    def latest(self):
        with self._lock:
            return self._seq, self._frame

    # -- recording
    def _writer_fps(self) -> float:
        for candidate in (self.fps_measured, self.fps_reported):
            if candidate and 1 <= candidate <= 120:
                return round(candidate, 2)
        return 25.0

    def start_recording(self) -> dict:
        with self._lock:
            if self._writer is not None:
                raise BadRequest("Already recording.")
            if self._frame is None:
                raise BadRequest("Stream has no frames yet.")
            h, w = self._frame.shape[:2]
            raw = storage.require_project(self.project) / "raw_media"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = storage.unique_path(raw, f"{self.name}_{stamp}.mp4")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self._writer_fps(), (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open the output file.")
            self._writer = writer
            self._writer_size = (w, h)
            self.recording_file = path.name
            self.recording_started = time.time()
            self.frames_written = 0
        return self.to_dict()

    def _close_writer(self) -> str | None:
        with self._lock:
            writer, filename = self._writer, self.recording_file
            self._writer = None
        if writer is not None:
            writer.release()
        return filename if writer is not None else None

    def stop_recording(self) -> dict:
        if self._writer is None:
            raise BadRequest("Not recording.")
        duration = time.time() - (self.recording_started or time.time())
        filename = self._close_writer()
        entry = _register_media(self.project, filename, "rtsp",
                                {"stream_url": self.url, "recorded_seconds": round(duration, 1)})
        self.recording_file = None
        self.recording_started = None
        return entry

    def snapshot(self, fmt: str = "jpg") -> dict:
        _, frame = self.latest()
        if frame is None:
            raise BadRequest("Stream has no frames yet.")
        out = storage.require_project(self.project) / "extracted_frames"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = storage.unique_path(out, f"{self.name}_snap_{stamp}.{fmt}")
        _write_image(path, frame, fmt)
        return {"image_id": f"extracted_frames/{path.name}"}

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def alive(self) -> bool:
        return not self._stop.is_set()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project": self.project,
            "url": self.url,
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps_measured, 1),
            "recording": self._writer is not None,
            "recording_file": self.recording_file,
            "recording_seconds": round(time.time() - self.recording_started, 1) if self.recording_started else 0,
            "frames_written": self.frames_written,
        }


class StreamManager:
    def __init__(self):
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def open(self, project: str, url: str, name: str) -> StreamSession:
        storage.require_project(project)
        if not url.strip():
            raise BadRequest("Stream URL is required.")
        session = StreamSession(project, url, name)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, stream_id: str) -> StreamSession:
        s = self._sessions.get(stream_id)
        if s is None:
            raise NotFound("Stream not found.")
        return s

    def for_project(self, project: str) -> list[StreamSession]:
        return [s for s in self._sessions.values() if s.project == project]

    def close(self, stream_id: str) -> None:
        with self._lock:
            s = self._sessions.pop(stream_id, None)
        if s is None:
            raise NotFound("Stream not found.")
        if s.recording_file:
            try:
                s.stop_recording()
            except Exception:  # noqa: BLE001
                pass
        s.stop()

    def close_project(self, project: str) -> None:
        for s in self.for_project(project):
            self.close(s.id)

    def close_all(self) -> None:
        for sid in list(self._sessions):
            try:
                self.close(sid)
            except Exception:  # noqa: BLE001
                pass


streams = StreamManager()


async def mjpeg_frames(session: StreamSession, max_width: int = 1280, fps_cap: float = 20.0):
    """Async generator yielding multipart MJPEG parts for an <img> tag."""
    last_seq = -1

    def encode(frame):
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None

    while session.alive:
        seq, frame = session.latest()
        if frame is not None and seq != last_seq:
            last_seq = seq
            jpg = await asyncio.to_thread(encode, frame)
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        await asyncio.sleep(1.0 / fps_cap)
