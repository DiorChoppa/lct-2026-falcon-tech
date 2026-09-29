"""Tag detection backend, swappable behind one interface.

NullBackend is the only implementation here: it returns no tags. Real
zero-shot detection (CLIP/OWLv2 against tagger/vocab.yaml prompts) needs
model weights that aren't available in this environment — see
tagger/README.md. Wiring (Enqueue -> queue -> worker -> gallery.SetTags) is
complete and tested against NullBackend; swapping in a real model is a
single class implementing TaggerBackend, no other code changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger("tagger")


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Tag:
    key: str
    confidence: float
    region: Region | None = None


class TaggerBackend(Protocol):
    def tag(self, crop: bytes) -> list[Tag]: ...


class NullBackend:
    """Placeholder: always returns no tags. Logs once so it's obvious in
    service logs that tagging is not actually happening yet.
    """

    def __init__(self) -> None:
        self._warned = False

    def tag(self, crop: bytes) -> list[Tag]:
        if not self._warned:
            log.warning(
                "NullBackend active: no zero-shot model loaded, all crops get zero tags"
            )
            self._warned = True
        return []
