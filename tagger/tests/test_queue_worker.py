import time

import pytest

from backend import Tag
from queue_worker import TaggerWorker


class RecordingBackend:
    def __init__(self, tags=None, error=False):
        self.calls = []
        self._tags = tags or []
        self._error = error

    def tag(self, crop: bytes):
        self.calls.append(crop)
        if self._error:
            raise RuntimeError("backend blew up")
        return self._tags


class RecordingGalleryClient:
    def __init__(self, error=False):
        self.calls = []
        self._error = error

    def set_tags(self, item_id: int, tags: list[Tag]) -> None:
        self.calls.append((item_id, tags))
        if self._error:
            raise RuntimeError("gallery unreachable")


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def worker_env():
    backend = RecordingBackend(tags=[Tag(key="roof_box", confidence=0.9)])
    gallery = RecordingGalleryClient()
    worker = TaggerWorker(backend, gallery)
    worker.start()
    yield worker, backend, gallery
    worker.stop()


def test_enqueue_tags_and_calls_gallery_set_tags(worker_env):
    worker, backend, gallery = worker_env

    worker.enqueue(item_id=42, crop=b"jpeg-bytes")

    assert wait_until(lambda: gallery.calls)
    assert backend.calls == [b"jpeg-bytes"]
    assert gallery.calls == [(42, [Tag(key="roof_box", confidence=0.9)])]


def test_backend_error_does_not_call_gallery_or_crash_worker():
    backend = RecordingBackend(error=True)
    gallery = RecordingGalleryClient()
    worker = TaggerWorker(backend, gallery)
    worker.start()

    worker.enqueue(item_id=1, crop=b"a")
    assert wait_until(lambda: backend.calls)
    time.sleep(0.05)

    assert gallery.calls == []
    worker.stop()


def test_gallery_error_is_caught_and_worker_keeps_processing():
    backend = RecordingBackend(tags=[])
    gallery = RecordingGalleryClient(error=True)
    worker = TaggerWorker(backend, gallery)
    worker.start()

    worker.enqueue(item_id=1, crop=b"a")
    assert wait_until(lambda: gallery.calls)

    worker.enqueue(item_id=2, crop=b"b")
    assert wait_until(lambda: len(backend.calls) == 2)
    worker.stop()


def test_stop_processes_no_further_jobs_after_join():
    backend = RecordingBackend()
    gallery = RecordingGalleryClient()
    worker = TaggerWorker(backend, gallery)
    worker.start()
    worker.stop()

    worker.enqueue(item_id=1, crop=b"a")
    time.sleep(0.05)

    assert backend.calls == []
