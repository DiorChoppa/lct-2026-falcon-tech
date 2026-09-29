"""In-memory queue + background worker: Enqueue returns immediately, tagging
and the gallery.SetTags callback happen on a separate thread. No broker
(NATS/Redis) — a crash drops only in-flight items, and gallery.Reindex(tags)
re-enqueues anything missed (04-architecture.md §6).
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from backend import TaggerBackend
from gallery_client import GalleryClient

log = logging.getLogger("tagger.worker")


@dataclass(frozen=True)
class Job:
    item_id: int
    crop: bytes


class TaggerWorker:
    def __init__(self, backend: TaggerBackend, gallery_client: GalleryClient) -> None:
        self._backend = backend
        self._gallery_client = gallery_client
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="tagger-worker")
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def enqueue(self, item_id: int, crop: bytes) -> None:
        self._queue.put(Job(item_id=item_id, crop=crop))

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            self._process(job)

    def _process(self, job: Job) -> None:
        try:
            tags = self._backend.tag(job.crop)
        except Exception:
            log.exception("tagging failed for item_id=%s", job.item_id)
            return
        try:
            self._gallery_client.set_tags(job.item_id, tags)
        except Exception:
            log.exception("gallery.SetTags failed for item_id=%s", job.item_id)
