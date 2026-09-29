"""End-to-end: real gRPC Enqueue call against a real TaggerService, backed by
a real (fake) Gallery gRPC server — proves the whole Enqueue -> queue ->
worker -> gallery.SetTags wiring works, not just its pieces in isolation.
"""

import time
from concurrent import futures

import gallery_pb2
import gallery_pb2_grpc
import grpc
import pytest
import tagger_pb2
import tagger_pb2_grpc

from backend import NullBackend, Tag
from gallery_client import GalleryClient
from queue_worker import TaggerWorker
from server import TaggerService


class FakeGalleryService(gallery_pb2_grpc.GalleryServicer):
    def __init__(self):
        self.set_tags_calls = []

    def SetTags(self, request, context):
        self.set_tags_calls.append(request)
        return gallery_pb2.SetTagsResponse()


class TaggingBackend:
    def tag(self, crop: bytes) -> list[Tag]:
        return [Tag(key="roof_box", confidence=0.75)]


def start_server(add_servicer, servicer):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_servicer(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


@pytest.fixture
def fake_gallery():
    servicer = FakeGalleryService()
    server, addr = start_server(gallery_pb2_grpc.add_GalleryServicer_to_server, servicer)
    yield servicer, addr
    server.stop(None)


@pytest.fixture
def tagger_stack(fake_gallery):
    _, gallery_addr = fake_gallery
    gallery_client = GalleryClient(gallery_addr)
    worker = TaggerWorker(TaggingBackend(), gallery_client)
    worker.start()
    server, addr = start_server(
        tagger_pb2_grpc.add_TaggerServicer_to_server, TaggerService(worker)
    )
    yield addr
    worker.stop()
    gallery_client.close()
    server.stop(None)


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_enqueue_reaches_gallery_set_tags_end_to_end(tagger_stack, fake_gallery):
    gallery_servicer, _ = fake_gallery
    channel = grpc.insecure_channel(tagger_stack)
    stub = tagger_pb2_grpc.TaggerStub(channel)

    response = stub.Enqueue(tagger_pb2.EnqueueRequest(item_id=7, crop=b"jpeg-bytes"))

    assert response == tagger_pb2.EnqueueResponse()
    assert wait_until(lambda: gallery_servicer.set_tags_calls)
    request = gallery_servicer.set_tags_calls[0]
    assert request.item_id == 7
    assert request.tags[0].key == "roof_box"
    assert abs(request.tags[0].confidence - 0.75) < 1e-6
    channel.close()


def test_enqueue_returns_immediately_even_with_null_backend(fake_gallery):
    _, gallery_addr = fake_gallery
    gallery_client = GalleryClient(gallery_addr)
    worker = TaggerWorker(NullBackend(), gallery_client)
    worker.start()
    server, addr = start_server(
        tagger_pb2_grpc.add_TaggerServicer_to_server, TaggerService(worker)
    )
    channel = grpc.insecure_channel(addr)
    stub = tagger_pb2_grpc.TaggerStub(channel)

    response = stub.Enqueue(tagger_pb2.EnqueueRequest(item_id=1, crop=b"x"), timeout=2)

    assert response == tagger_pb2.EnqueueResponse()
    channel.close()
    worker.stop()
    gallery_client.close()
    server.stop(None)
