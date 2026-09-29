"""gRPC client for gallery.SetTags — the only way tagger's results reach the
database; tagger itself never touches Postgres or S3 (04-architecture.md §6).
"""

from __future__ import annotations

import logging

import gallery_pb2  # сгенерировано `just proto-py`
import gallery_pb2_grpc
import grpc
import tagger_pb2

from backend import Tag

log = logging.getLogger("tagger.gallery_client")


class GalleryClient:
    def __init__(self, addr: str) -> None:
        self._addr = addr
        self._channel: grpc.Channel | None = None
        self._stub: gallery_pb2_grpc.GalleryStub | None = None

    def _stub_for(self) -> gallery_pb2_grpc.GalleryStub:
        if self._stub is None:
            self._channel = grpc.insecure_channel(self._addr)
            self._stub = gallery_pb2_grpc.GalleryStub(self._channel)
        return self._stub

    def set_tags(self, item_id: int, tags: list[Tag]) -> None:
        request = gallery_pb2.SetTagsRequest(
            item_id=item_id,
            tags=[_to_proto_tag(t) for t in tags],
        )
        self._stub_for().SetTags(request, timeout=5)

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None


def _to_proto_tag(tag: Tag) -> tagger_pb2.Tag:
    kwargs = {"key": tag.key, "confidence": tag.confidence}
    if tag.region is not None:
        kwargs["region"] = tagger_pb2.Region(
            x=tag.region.x, y=tag.region.y, w=tag.region.w, h=tag.region.h
        )
    return tagger_pb2.Tag(**kwargs)
