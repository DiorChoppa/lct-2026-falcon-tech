"""gRPC-сервер Tagger (proto/reid/v1/tagger.proto).

Enqueue кладёт задачу в очередь и сразу отвечает — теги считаются и
отправляются в gallery.SetTags на фоновом потоке (queue_worker.py).
Стабы *_pb2*.py генерируются `just proto-py` и в git не хранятся.
"""

from __future__ import annotations

import logging
import os
from concurrent import futures

import grpc
import tagger_pb2
import tagger_pb2_grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from backend import NullBackend
from gallery_client import GalleryClient
from queue_worker import TaggerWorker

log = logging.getLogger("tagger")


class TaggerService(tagger_pb2_grpc.TaggerServicer):
    def __init__(self, worker: TaggerWorker) -> None:
        self._worker = worker

    def Enqueue(self, request, context):
        self._worker.enqueue(request.item_id, request.crop)
        return tagger_pb2.EnqueueResponse()


def serve() -> None:
    logging.basicConfig(level=logging.INFO)
    addr = os.environ.get("TAGGER_ADDR", "0.0.0.0:50052")
    gallery_addr = os.environ.get("GALLERY_ADDR", "gallery:50054")

    gallery_client = GalleryClient(gallery_addr)
    worker = TaggerWorker(NullBackend(), gallery_client)
    worker.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tagger_pb2_grpc.add_TaggerServicer_to_server(TaggerService(worker), server)
    health_servicer = health.HealthServicer()
    health_servicer.set("reid.v1.Tagger", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    server.add_insecure_port(addr)
    log.info("tagger listening on %s, gallery at %s", addr, gallery_addr)
    server.start()
    try:
        server.wait_for_termination()
    finally:
        worker.stop()
        gallery_client.close()


if __name__ == "__main__":
    serve()
