"""YOLOFlow Studio - FastAPI backend.

Run from the backend/ directory:
    uvicorn main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import storage
from core.catalog import catalog_payload
from core.jobs import TERMINAL, jobs
from core.storage import BadRequest, NotFound
from services import annotation_service, inference_service, kaggle_service, media_service, model_registry, training_service
from services.inference_service import live
from services.media_service import streams


@asynccontextmanager
async def lifespan(_app: FastAPI):
    storage.init_workspace()
    training_service.resume_all()
    yield
    live.stop_all()
    streams.close_all()


app = FastAPI(title="YOLOFlow Studio", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(NotFound)
async def _not_found(_req: Request, exc: NotFound):
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(BadRequest)
async def _bad_request(_req: Request, exc: BadRequest):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(Exception)
async def _unhandled(_req: Request, exc: Exception):
    traceback.print_exception(exc)
    return JSONResponse({"detail": f"{type(exc).__name__}: {str(exc)[:500]}"}, status_code=500)


@app.exception_handler(kaggle_service.KaggleError)
async def _kaggle_error(_req: Request, exc: kaggle_service.KaggleError):
    return JSONResponse({"detail": str(exc)}, status_code=502)


# ================================================================ schemas

class ProjectIn(BaseModel):
    name: str


class ClassIn(BaseModel):
    name: str
    color: Optional[str] = None


class ClassesBulkIn(BaseModel):
    names: list[str]


class ClassPatch(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None


class Box(BaseModel):
    cls: int
    cx: float
    cy: float
    w: float
    h: float


class AnnotationIn(BaseModel):
    boxes: list[Box]


class AutoLabelIn(BaseModel):
    image_id: str
    model: str = "base:yolo11n.pt"
    conf: float = Field(0.25, ge=0.01, le=1.0)
    iou: float = Field(0.5, ge=0.05, le=0.95)
    classes: Optional[list[str]] = None


class ExtractIn(BaseModel):
    mode: str = "fps"  # "fps" | "interval"
    fps: float = 1.0
    interval: int = 30
    format: str = "jpg"
    quality: int = Field(95, ge=50, le=100)
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


class StreamIn(BaseModel):
    url: str
    name: str = "stream"


class KaggleCredsIn(BaseModel):
    username: str
    key: str


class KaggleVerifyIn(BaseModel):
    username: Optional[str] = None
    key: Optional[str] = None


class LiveIn(BaseModel):
    url: str
    model: str
    conf: float = Field(0.25, ge=0.01, le=1.0)
    iou: float = Field(0.5, ge=0.05, le=0.95)
    imgsz: Optional[int] = None
    classes: Optional[list[str]] = None


class LivePatch(BaseModel):
    conf: Optional[float] = Field(None, ge=0.01, le=1.0)
    iou: Optional[float] = Field(None, ge=0.05, le=0.95)
    classes: Optional[list[str]] = None


class RunIn(BaseModel):
    backend: str = "kaggle"
    config: dict = {}


# ================================================================ meta & config

@app.get("/api/health")
def health():
    return {"ok": True, "time": time.time()}


@app.get("/api/meta/catalog")
def catalog():
    return catalog_payload()


def _public_config() -> dict:
    cfg = storage.load_config()
    key = cfg["kaggle"].get("key", "")
    cfg["kaggle"] = {**cfg["kaggle"], "key": "", "has_key": bool(key),
                     "key_hint": f"…{key[-4:]}" if len(key) >= 8 else ""}
    return cfg


@app.get("/api/config")
def get_config():
    return _public_config()


@app.put("/api/config/kaggle")
def set_kaggle(body: KaggleCredsIn):
    def fn(cfg):
        cfg["kaggle"]["username"] = body.username.strip()
        if body.key.strip():  # blank key = keep the stored one
            cfg["kaggle"]["key"] = body.key.strip()
        cfg["kaggle"]["verified"] = False

    storage.update_config(fn)
    return _public_config()


@app.post("/api/config/kaggle/verify")
def verify_kaggle(body: KaggleVerifyIn):
    result = kaggle_service.verify(body.username or None, body.key or None)
    if result["ok"] and not body.key:
        def fn(cfg):
            cfg["kaggle"]["verified"] = True
            cfg["kaggle"]["verified_at"] = time.time()

        storage.update_config(fn)
    return result


@app.put("/api/config/defaults")
def set_defaults(body: dict):
    cfg = training_service.normalize_config(body)
    storage.update_config(lambda c: c["training_defaults"].update(cfg))
    return _public_config()


# ================================================================ projects

def _project_summary(meta: dict) -> dict:
    name = meta["name"]
    root = storage.project_dir(name)
    counts = annotation_service.split_counts(name)
    return {
        "name": name,
        "created_at": meta.get("created_at"),
        "classes": meta.get("classes", []),
        "media_count": sum(1 for p in (root / "raw_media").glob("*") if p.is_file()),
        "frame_count": sum(1 for p in (root / "extracted_frames").glob("*") if p.is_file()),
        "labeled_count": counts["train"] + counts["val"],
    }


@app.get("/api/projects")
def projects():
    return [_project_summary(m) for m in storage.list_projects()]


@app.post("/api/projects")
def create_project(body: ProjectIn):
    return _project_summary(storage.create_project(body.name))


@app.get("/api/projects/{project}")
def get_project(project: str):
    return _project_summary(storage.load_meta(project))


@app.delete("/api/projects/{project}")
def delete_project(project: str):
    streams.close_project(project)
    live.stop_project(project)
    storage.delete_project(project)
    return {"ok": True}


@app.get("/api/projects/{project}/stats")
def project_stats(project: str):
    summary = _project_summary(storage.load_meta(project))
    runs = training_service.list_runs(project)
    latest = next((r for r in runs if r.get("status") == "completed"), None)
    return {**summary, "dataset": annotation_service.dataset_stats(project),
            "runs_total": len(runs), "latest_run": latest}


# ================================================================ classes

@app.post("/api/projects/{project}/classes")
def add_class(project: str, body: ClassIn):
    return annotation_service.add_classes(project, [body.name], body.color)


@app.post("/api/projects/{project}/classes/bulk")
def add_classes_bulk(project: str, body: ClassesBulkIn):
    return annotation_service.add_classes(project, body.names)


@app.patch("/api/projects/{project}/classes/{class_id}")
def patch_class(project: str, class_id: int, body: ClassPatch):
    return annotation_service.update_class(project, class_id, body.name, body.color)


@app.delete("/api/projects/{project}/classes/{class_id}")
def delete_class(project: str, class_id: int):
    return annotation_service.delete_class(project, class_id)


# ================================================================ media

@app.get("/api/projects/{project}/media")
def list_media(project: str):
    return media_service.list_media(project)


@app.post("/api/projects/{project}/media")
async def upload_media(project: str, files: list[UploadFile] = File(...)):
    storage.require_project(project)
    saved, errors = [], []
    for f in files:
        try:
            saved.append(await media_service.save_upload(project, f))
        except BadRequest as exc:
            errors.append(str(exc))
    return {"saved": saved, "errors": errors}


@app.delete("/api/projects/{project}/media/{filename}")
def delete_media(project: str, filename: str):
    media_service.delete_media(project, filename)
    return {"ok": True}


@app.post("/api/projects/{project}/media/{filename}/extract")
def extract(project: str, filename: str, body: ExtractIn):
    storage.require_project(project)
    if body.mode not in ("fps", "interval"):
        raise BadRequest("mode must be 'fps' or 'interval'.")
    job = jobs.start(
        "extract", project,
        lambda j: media_service.extract_frames(j, project, filename, body.mode, body.fps, body.interval,
                                               body.format, body.quality, body.start_sec, body.end_sec),
        meta={"filename": filename},
    )
    return job.to_dict()


@app.get("/api/projects/{project}/files/{folder}/{filename}")
def get_file(project: str, folder: str, filename: str):
    return FileResponse(storage.project_file(project, folder, filename))


@app.get("/api/projects/{project}/thumbs/{folder}/{filename}")
def get_thumb(project: str, folder: str, filename: str):
    return FileResponse(media_service.thumbnail(project, folder, filename),
                        headers={"Cache-Control": "max-age=3600"})


# ================================================================ jobs (SSE progress)

@app.get("/api/jobs")
def list_jobs(project: Optional[str] = None):
    return [j.to_dict() for j in jobs.list(project)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise NotFound("Job not found.")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise NotFound("Job not found.")
    job.cancel()
    return job.to_dict()


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    job = jobs.get(job_id)
    if not job:
        raise NotFound("Job not found.")

    async def gen():
        last = None
        while True:
            if await request.is_disconnected():
                return
            payload = json.dumps(job.to_dict())
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if job.status in TERMINAL:
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ================================================================ images & annotations

@app.get("/api/projects/{project}/images")
def list_images(project: str):
    return annotation_service.list_images(project)


@app.delete("/api/projects/{project}/images/{folder}/{filename}")
def delete_image(project: str, folder: str, filename: str):
    annotation_service.delete_image(project, folder, filename)
    if folder == "raw_media":
        storage.update_meta(project, lambda m: m["media"].pop(filename, None))
    return {"ok": True}


@app.get("/api/projects/{project}/annotations/{folder}/{filename}")
def get_annotation(project: str, folder: str, filename: str):
    return annotation_service.get_annotation(project, f"{folder}/{filename}")


@app.put("/api/projects/{project}/annotations/{folder}/{filename}")
def put_annotation(project: str, folder: str, filename: str, body: AnnotationIn):
    return annotation_service.save_annotation(project, f"{folder}/{filename}", [b.model_dump() for b in body.boxes])


@app.delete("/api/projects/{project}/annotations/{folder}/{filename}")
def delete_annotation(project: str, folder: str, filename: str):
    annotation_service.delete_annotation(project, f"{folder}/{filename}")
    return {"ok": True}


@app.get("/api/projects/{project}/autolabel/models")
def autolabel_models(project: str):
    return annotation_service.list_autolabel_models(project)


@app.get("/api/projects/{project}/autolabel/classes")
def autolabel_classes(project: str, model: str):
    return annotation_service.model_classes(project, model)


@app.post("/api/projects/{project}/autolabel")
def autolabel(project: str, body: AutoLabelIn):
    return annotation_service.autolabel(project, body.image_id, body.model, body.conf, body.classes, body.iou)


# ================================================================ RTSP streams

@app.get("/api/projects/{project}/streams")
def project_streams(project: str):
    return [s.to_dict() for s in streams.for_project(project)]


@app.post("/api/projects/{project}/streams")
def open_stream(project: str, body: StreamIn):
    return streams.open(project, body.url, body.name).to_dict()


@app.get("/api/streams/{stream_id}")
def stream_status(stream_id: str):
    return streams.get(stream_id).to_dict()


@app.get("/api/streams/{stream_id}/mjpeg")
def stream_mjpeg(stream_id: str):
    session = streams.get(stream_id)
    return StreamingResponse(media_service.mjpeg_frames(session),
                             media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache, no-store"})


@app.post("/api/streams/{stream_id}/record/start")
def record_start(stream_id: str):
    return streams.get(stream_id).start_recording()


@app.post("/api/streams/{stream_id}/record/stop")
def record_stop(stream_id: str):
    return streams.get(stream_id).stop_recording()


@app.post("/api/streams/{stream_id}/snapshot")
def snapshot(stream_id: str):
    return streams.get(stream_id).snapshot()


@app.delete("/api/streams/{stream_id}")
def close_stream(stream_id: str):
    streams.close(stream_id)
    return {"ok": True}


# ================================================================ models (trained / uploaded / COCO)

@app.get("/api/models")
def models(project: Optional[str] = None):
    return model_registry.list_models(project)


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...), name: str = Form("")):
    return await model_registry.save_upload(file, name)


@app.delete("/api/models/custom/{filename}")
def delete_model(filename: str):
    model_registry.delete_custom(filename)
    return {"ok": True}


# ================================================================ inference

def _parse_classes(raw: str) -> Optional[list[str]]:
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BadRequest("classes must be a JSON list.") from exc
    return [str(v) for v in val] or None


def _imgsz(v: int) -> Optional[int]:
    if not v:
        return None
    if v < 32 or v % 32:
        raise BadRequest("Image size must be a multiple of 32.")
    return v


@app.post("/api/projects/{project}/inference/image")
async def infer_image(project: str, model: str = Form(...), conf: float = Form(0.25), iou: float = Form(0.5),
                      imgsz: int = Form(0), classes: str = Form(""), image_id: str = Form(""),
                      file: Optional[UploadFile] = File(None)):
    storage.require_project(project)
    data = await file.read() if file is not None else None
    return await asyncio.to_thread(inference_service.detect_image, project, model, conf, iou, _imgsz(imgsz),
                                   _parse_classes(classes), data, image_id or None)


@app.post("/api/projects/{project}/inference/video")
async def infer_video(project: str, model: str = Form(...), conf: float = Form(0.25), iou: float = Form(0.5),
                      imgsz: int = Form(0), classes: str = Form(""), media_filename: str = Form(""),
                      file: Optional[UploadFile] = File(None)):
    storage.require_project(project)
    model_registry.resolve(project, model)
    upload_path = await inference_service.save_video_upload(project, file) if file is not None else None
    src = inference_service.resolve_video_source(project, media_filename or None, upload_path)
    size, cls = _imgsz(imgsz), _parse_classes(classes)
    job = jobs.start(
        "inference_video", project,
        lambda j: inference_service.run_video(j, project, src, model, conf, iou, size, cls, upload_path is not None),
        meta={"source": src.name, "model": model},
    )
    return job.to_dict()


@app.get("/api/projects/{project}/inference/videos")
def infer_video_results(project: str):
    return inference_service.list_video_results(project)


@app.get("/api/projects/{project}/inference/videos/{result_id}/file")
def infer_video_file(project: str, result_id: str):
    return FileResponse(inference_service.video_result_file(project, result_id))


@app.delete("/api/projects/{project}/inference/videos/{result_id}")
def infer_video_delete(project: str, result_id: str):
    inference_service.delete_video_result(project, result_id)
    return {"ok": True}


@app.get("/api/projects/{project}/inference/live")
def live_sessions(project: str):
    return [s.to_dict() for s in live.for_project(project)]


@app.post("/api/projects/{project}/inference/live")
def live_start(project: str, body: LiveIn):
    session = live.start(project, body.url, body.model, body.conf, body.iou, _imgsz(body.imgsz or 0), body.classes)
    return session.to_dict()


@app.get("/api/inference/live/{sid}")
def live_status(sid: str):
    return live.get(sid).to_dict()


@app.patch("/api/inference/live/{sid}")
def live_update(sid: str, body: LivePatch):
    s = live.get(sid)
    s.update(body.conf, body.iou, body.classes)
    return s.to_dict()


@app.get("/api/inference/live/{sid}/mjpeg")
def live_mjpeg(sid: str):
    return StreamingResponse(media_service.mjpeg_frames(live.get(sid)),
                             media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache, no-store"})


@app.delete("/api/inference/live/{sid}")
def live_stop(sid: str):
    live.stop(sid)
    return {"ok": True}


# ================================================================ training runs

@app.get("/api/projects/{project}/runs")
def runs(project: str):
    return training_service.list_runs(project)


@app.post("/api/projects/{project}/runs")
def create_run(project: str, body: RunIn):
    return training_service.create_run(project, body.backend, body.config)


@app.get("/api/projects/{project}/runs/{run_id}")
def get_run(project: str, run_id: str):
    return training_service.load_run(project, run_id)


@app.get("/api/projects/{project}/runs/{run_id}/log")
def run_log(project: str, run_id: str):
    return {"log": training_service.read_log(project, run_id)}


@app.get("/api/projects/{project}/runs/{run_id}/results")
def run_results(project: str, run_id: str):
    return training_service.results_table(project, run_id)


@app.get("/api/projects/{project}/runs/{run_id}/artifacts/{name}")
def run_artifact(project: str, run_id: str, name: str):
    path = training_service.artifact_path(project, run_id, name)
    download = path.suffix in (".pt", ".onnx", ".csv", ".yaml", ".json")
    return FileResponse(path, filename=f"{run_id}_{name}" if download else None)


@app.post("/api/projects/{project}/runs/{run_id}/cancel")
def cancel_run(project: str, run_id: str):
    return training_service.cancel_run(project, run_id)


@app.post("/api/projects/{project}/runs/{run_id}/refresh")
def refresh_run(project: str, run_id: str):
    return training_service.refresh_run(project, run_id)


@app.delete("/api/projects/{project}/runs/{run_id}")
def delete_run(project: str, run_id: str):
    training_service.delete_run(project, run_id)
    return {"ok": True}


# ================================================================ static frontend (production build)

_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise NotFound("Unknown API route.")
        candidate = (_DIST / full_path).resolve()
        if full_path and candidate.is_file() and _DIST in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
