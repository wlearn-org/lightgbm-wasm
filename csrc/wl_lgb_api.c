/*
 * wl_lgb_api.c -- Thin C glue layer for LightGBM WASM port.
 *
 * Wraps LightGBM C API functions that use int64_t parameters with int32
 * equivalents, since Emscripten legalizes i64 to (lo32, hi32) pairs at
 * the JS boundary.
 *
 * All data flows through float32 (C_API_DTYPE_FLOAT32) to match JS
 * Float32Array. Labels use float32 (per LightGBM C API requirement).
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <LightGBM/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Error ---- */

const char* wl_lgb_get_last_error(void) {
  return LGBM_GetLastError();
}

/* ---- Dataset ---- */

int wl_lgb_dataset_create_from_mat(const float* data, int nrow, int ncol,
                                   const char* params, void** out) {
  return LGBM_DatasetCreateFromMat(
    (const void*)data,
    C_API_DTYPE_FLOAT32,
    (int32_t)nrow,
    (int32_t)ncol,
    1,        /* is_row_major = true */
    params,
    NULL,     /* no reference dataset */
    (DatasetHandle*)out
  );
}

int wl_lgb_dataset_set_field(void* handle, const char* field,
                             const void* data, int n, int type) {
  return LGBM_DatasetSetField(
    (DatasetHandle)handle,
    field,
    data,
    n,
    type
  );
}

int wl_lgb_dataset_free(void* handle) {
  return LGBM_DatasetFree((DatasetHandle)handle);
}

/* ---- Booster ---- */

int wl_lgb_booster_create(void* train_data, const char* params, void** out) {
  return LGBM_BoosterCreate(
    (DatasetHandle)train_data,
    params,
    (BoosterHandle*)out
  );
}

int wl_lgb_booster_update(void* handle, int* is_finished) {
  return LGBM_BoosterUpdateOneIter(
    (BoosterHandle)handle,
    is_finished
  );
}

int wl_lgb_booster_get_num_classes(void* handle, int* out) {
  return LGBM_BoosterGetNumClasses(
    (BoosterHandle)handle,
    out
  );
}

int wl_lgb_booster_free(void* handle) {
  return LGBM_BoosterFree((BoosterHandle)handle);
}

/* ---- Predict (int64 -> int32 wrapper) ---- */

int wl_lgb_booster_predict(void* handle, const float* data,
                           int nrow, int ncol, int predict_type,
                           int num_iteration, const char* params,
                           int* out_len, double* out_result) {
  int64_t len64 = 0;
  int ret = LGBM_BoosterPredictForMat(
    (BoosterHandle)handle,
    (const void*)data,
    C_API_DTYPE_FLOAT32,
    (int32_t)nrow,
    (int32_t)ncol,
    1,              /* is_row_major = true */
    predict_type,
    0,              /* start_iteration = 0 (all) */
    num_iteration,  /* <= 0 means no limit */
    params,
    &len64,
    out_result
  );
  if (out_len) *out_len = (int)len64;
  return ret;
}

/* ---- Save model (int64 -> int32 wrapper) ---- */

int wl_lgb_booster_save_model(void* handle, int buffer_len,
                              int* out_len, char* out_str) {
  int64_t len64 = 0;
  int ret = LGBM_BoosterSaveModelToString(
    (BoosterHandle)handle,
    0,              /* start_iteration = 0 */
    0,              /* num_iteration = 0 (all) */
    0,              /* feature_importance_type = split */
    (int64_t)buffer_len,
    &len64,
    out_str
  );
  if (out_len) *out_len = (int)len64;
  return ret;
}

/* ---- Load model ---- */

int wl_lgb_booster_load_model(const char* model_str,
                              int* out_num_iterations, void** out) {
  return LGBM_BoosterLoadModelFromString(
    model_str,
    out_num_iterations,
    (BoosterHandle*)out
  );
}

#ifdef __cplusplus
}
#endif
