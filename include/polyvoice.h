/* polyvoice.h — C FFI ABI v3 for polyvoice Pipeline (v0.6.5+).
 * Threading: PolyvoicePipeline is Send. Each handle must be destroyed exactly once.
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
    POLYVOICE_ERR_AUDIO_TOO_LONG = 3,
    POLYVOICE_ERR_MODEL_LOAD = 10,
    POLYVOICE_ERR_INFERENCE = 11,
    POLYVOICE_ERR_OUT_OF_MEMORY = 20,
    POLYVOICE_ERR_REGISTRY = 30,
    POLYVOICE_ERR_INTERNAL = 99
} polyvoice_status_t;

/** Create a pipeline from a profile. */
int polyvoice_pipeline_create(polyvoice_profile_t profile,
                              const char* models_cache_dir,
                              PolyvoicePipeline** out_handle);

/**
 * Run diarization on f32 samples. Returns JSON via out_json.
 * Must not be called concurrently with another run or destroy on the same handle.
 */
int polyvoice_pipeline_run(PolyvoicePipeline* pipeline,
                           const float* samples,
                           size_t n_samples,
                           uint32_t sample_rate,
                           char** out_json,
                           size_t* out_json_len);

/**
 * Destroy a pipeline. Must be called exactly once per handle.
 * Must not be called concurrently with run on the same handle.
 */
void polyvoice_pipeline_destroy(PolyvoicePipeline* pipeline);

/** Free a JSON string returned by polyvoice_pipeline_run. */
void polyvoice_free_string(char* p, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* POLYVOICE_H */
