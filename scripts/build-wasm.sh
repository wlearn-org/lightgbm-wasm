#!/bin/bash
set -euo pipefail

# Build LightGBM v4.6.0 as WASM via Emscripten
# Prerequisites: emsdk activated (emcc, emcmake, emmake in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/lightgbm"
BUILD_DIR="${PROJECT_DIR}/build"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source ~/tools/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/CMakeLists.txt" ]; then
  echo "ERROR: LightGBM upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init --recursive"
  exit 1
fi

echo "=== Applying patches ==="
if [ -d "${PROJECT_DIR}/patches" ] && ls "${PROJECT_DIR}/patches"/*.patch &> /dev/null 2>&1; then
  for patch in "${PROJECT_DIR}/patches"/*.patch; do
    echo "Applying: $(basename "$patch")"
    (cd "$UPSTREAM_DIR" && git apply --check "$patch" 2>/dev/null && git apply "$patch") || \
      echo "  (already applied or not applicable)"
  done
else
  echo "  No patches found"
fi

echo "=== CMake configure ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

emcmake cmake "$UPSTREAM_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_OPENMP=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_GPU=OFF \
  -DUSE_MPI=OFF \
  -DUSE_HDFS=OFF \
  -DUSE_SWIG=OFF \
  -DBUILD_STATIC_LIB=ON \
  -DBUILD_CLI=OFF \
  -DCMAKE_C_FLAGS="-O2 -fexceptions" \
  -DCMAKE_CXX_FLAGS="-O2 -fexceptions"

echo "=== Building ==="
emmake make -j"$(nproc)" 2>&1

echo "=== Linking WASM ==="
mkdir -p "$OUTPUT_DIR"

# Find the static library (may be in build/ or upstream/lightgbm/)
LGB_LIB=$(find "$BUILD_DIR" "$UPSTREAM_DIR" -name 'lib_lightgbm.a' -print -quit)

if [ -z "$LGB_LIB" ]; then
  echo "ERROR: lib_lightgbm.a not found"
  echo "Looking for any .a files:"
  find "$BUILD_DIR" "$UPSTREAM_DIR" -name '*.a' -print
  exit 1
fi

echo "Using: $LGB_LIB"

EXPORTED_FUNCTIONS='[
  "_wl_lgb_get_last_error",
  "_wl_lgb_dataset_create_from_mat",
  "_wl_lgb_dataset_set_field",
  "_wl_lgb_dataset_free",
  "_wl_lgb_booster_create",
  "_wl_lgb_booster_update",
  "_wl_lgb_booster_get_num_classes",
  "_wl_lgb_booster_free",
  "_wl_lgb_booster_predict",
  "_wl_lgb_booster_save_model",
  "_wl_lgb_booster_load_model",
  "_malloc",
  "_free"
]'

# Remove newlines for emcc
EXPORTED_FUNCTIONS=$(echo "$EXPORTED_FUNCTIONS" | tr -d '\n' | tr -s ' ')

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF32","HEAPF64","HEAPU8","HEAP32","UTF8ToString"]'

em++ \
  "${PROJECT_DIR}/csrc/wl_lgb_api.c" \
  -I "$UPSTREAM_DIR/include" \
  -Wl,--whole-archive "$LGB_LIB" -Wl,--no-whole-archive \
  -O2 \
  -fexceptions \
  -o "${OUTPUT_DIR}/lightgbm.cjs" \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s EXPORT_NAME=createLightGBM \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s ENVIRONMENT='web,node' \
  -s FORCE_FILESYSTEM=0

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: LightGBM v4.6.0
upstream_commit: $(cd "$UPSTREAM_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -fexceptions SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/lightgbm.cjs"
cat "${OUTPUT_DIR}/BUILD_INFO"
