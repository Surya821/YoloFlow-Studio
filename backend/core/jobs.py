"""Minimal in-memory background job registry with progress reporting (consumed via SSE)."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Callable

TERMINAL = {"completed", "failed", "cancelled"}


class JobCancelled(Exception):
    pass


class Job:
    def __init__(self, kind: str, project: str, meta: dict | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.project = project
        self.meta = meta or {}
        self.status = "running"
        self.current = 0
        self.total = 0
        self.message = ""
        self.result: dict | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self.updated_at = self.created_at
        self._cancel = threading.Event()

    def update(self, current: int | None = None, total: int | None = None, message: str | None = None):
        if current is not None:
            self.current = current
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message
        self.updated_at = time.time()

    def check_cancelled(self):
        if self._cancel.is_set():
            raise JobCancelled()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()

    def to_dict(self) -> dict:
        progress = (self.current / self.total) if self.total else (1.0 if self.status == "completed" else 0.0)
        return {
            "id": self.id,
            "kind": self.kind,
            "project": self.project,
            "meta": self.meta,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "progress": min(1.0, progress),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, project: str, target: Callable[[Job], dict | None], meta: dict | None = None) -> Job:
        job = Job(kind, project, meta)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()

        def runner():
            try:
                job.result = target(job) or {}
                job.status = "cancelled" if job.cancelled else "completed"
            except JobCancelled:
                job.status = "cancelled"
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                job.error = str(exc)
                job.status = "failed"
            job.updated_at = time.time()

        threading.Thread(target=runner, daemon=True, name=f"job-{kind}-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, project: str | None = None) -> list[Job]:
        return [j for j in self._jobs.values() if project is None or j.project == project]

    def _prune(self):
        cutoff = time.time() - 6 * 3600
        for jid in [j.id for j in self._jobs.values() if j.status in TERMINAL and j.updated_at < cutoff]:
            self._jobs.pop(jid, None)


jobs = JobManager()
