"""Set native library paths, then replace this process with the offline Rust driver."""
import importlib.util
import os
from pathlib import Path
import sys

spec = importlib.util.find_spec("onnxruntime")
if spec is None or spec.origin is None:
    raise SystemExit("ONNX Runtime is not installed in image")
capi = Path(spec.origin).parent / "capi"
library = capi / "libonnxruntime.so.1.24.4"
if not library.is_file():
    raise SystemExit(f"Pinned ORT library missing: {library}")
paths = [str(capi), *(str(path) for path in sorted((capi.parent.parent / "nvidia").glob("*/lib")))]
if os.environ.get("LD_LIBRARY_PATH"):
    paths.append(os.environ["LD_LIBRARY_PATH"])
os.environ["LD_LIBRARY_PATH"] = ":".join(paths)
os.environ["ORT_DYLIB_PATH"] = str(library)
os.execvpe("lct-offline", ["lct-offline", *sys.argv[1:]], os.environ)
