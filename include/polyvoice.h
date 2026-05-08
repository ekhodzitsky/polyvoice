/* polyvoice.h — C FFI v2 ABI (M6b).
 * v1.0 architecture: profile-based Pipeline. Old ABI removed.
 */
#ifndef POLYVOICE_H
#define POLYVOICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PolyvoicePipeline PolyvoicePipeline;

typedef enum {
    POLYVOICE_PROFILE_MOBILE = 0,
    POLYVOICE_PROFILE_BALANCED = 1
} polyvoice_profile_t;

typedef enum {
    POLYVOICE_OK = 0,
    POLYVOICE_ERR_INVALID_ARG = 1,
    POLYVOICE_ERR_AUDIO_TOO_SHORT = 2,
    POLYVOICE_ERR_MODEL_LOAD = 10,
    POLYVOICE_ERR_INFERENCE = 11,
    POLYVOICE_ERR_OUT_OF_MEMORY = 20,
    POLYVOICE_ERR_REGISTRY = 30,
    POLYVOICE_ERR_INTERNAL = 99
} polyvoice_status_t;

int polyvoice_pipeline_create(polyvoice_profile_t profile,
                              const char* models_cache_dir,
                              PolyvoicePipeline** out_handle);

int polyvoice_pipeline_run(PolyvoicePipeline* pipeline,
                           const float* samples,
                           size_t n_samples,
                           uint32_t sample_rate,
                           char** out_json,
                           size_t* out_json_len);

void polyvoice_pipeline_destroy(PolyvoicePipeline* pipeline);
void polyvoice_free_string(char* p, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* POLYVOICE_H */
