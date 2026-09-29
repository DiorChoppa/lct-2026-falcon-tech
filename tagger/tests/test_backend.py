from backend import NullBackend


def test_null_backend_returns_no_tags():
    backend = NullBackend()

    assert backend.tag(b"not really a crop") == []


def test_null_backend_warns_once(caplog):
    import logging

    backend = NullBackend()
    with caplog.at_level(logging.WARNING):
        backend.tag(b"a")
        backend.tag(b"b")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
