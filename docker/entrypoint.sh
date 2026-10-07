#!/bin/sh
# Make the CUDA libraries that came with the torch wheels visible to ctranslate2 (faster-whisper).
NV_LIBS=$(python - <<'PY'
import importlib.util
dirs = []
for mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib", "nvidia.cuda_runtime.lib", "nvidia.cuda_nvrtc.lib"):
    try:
        spec = importlib.util.find_spec(mod)
    except ModuleNotFoundError:
        spec = None
    if spec and spec.submodule_search_locations:
        dirs.extend(spec.submodule_search_locations)
print(":".join(dirs))
PY
)
[ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="$NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "${PORTAL_DATA_ROOT:-/data}" "${PORTAL_MODELS_DIR:-/models}" 2>/dev/null || true
exec "$@"
