# Upstream: LightGBM

## Source

- **Repository:** https://github.com/microsoft/LightGBM
- **License:** MIT
- **Component used:** Full LightGBM library (C++ core + C API)

## Tracked version

- **Tag:** v4.6.0
- **Submodule path:** `upstream/lightgbm/`

## Modifications for WASM

1. **No threading:** Single-threaded WASM build. `USE_OPENMP=OFF`, no `-pthread` flag.

2. **No GPU/CUDA:** `USE_CUDA=OFF`, `USE_GPU=OFF`. WASM targets CPU only.

3. **No distributed:** `USE_MPI=OFF`, `USE_HDFS=OFF`. Single-process execution.

4. **No CLI:** `BUILD_CLI=OFF`. Only the library is built.

5. **C++ exceptions enabled:** `-fexceptions` flag required because LightGBM uses C++
   exceptions internally. Without this flag, Emscripten silently drops exceptions.

6. **int64 boundary handling:** LightGBM C API uses `int64_t` for buffer lengths in
   `LGBM_BoosterPredictForMat` and `LGBM_BoosterSaveModelToString`. The C glue layer
   (`csrc/wl_lgb_api.c`) provides int32 wrappers for all functions called from JS,
   since Emscripten legalizes i64 to (lo32, hi32) pairs at the JS boundary.

7. **Model format:** LightGBM text format inside WLRN bundle. Cross-language compatible
   with Python `wlearn.lightgbm` wrapper.

## Build output

- `wasm/lightgbm.cjs` (WASM embedded via SINGLE_FILE=1)
- Flags: `-O2 -fexceptions`
- Single-threaded, CPU only

## Update policy

- Track stable tags (not main branch)
- Apply patches via `patches/` directory if needed
- Verify golden test fixtures after each update
- Run both JS and Python cross-language tests
