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
    /* Reserved for ABI stability: never returned by the current implementation
     * (pipeline v2 has no matching error). Do not reuse the value 2. */
    POLYVOICE_ERR_AUDIO_TOO_SHORT = 2,
    POLYVOICE_ERR_AUDIO_TOO_LONG = 3,
    POLYVOICE_ERR_MODEL_LOAD = 10,
    POLYVOICE_ERR_INFERENCE = 11,
    POLYVOICE_ERR_OUT_OF_MEMORY = 20,
    POLYVOICE_ERR_REGISTRY = 30,
    POLYVOICE_ERR_INTERNAL = 99
} polyvoice_status_t;

typedef enum {
    POLYVOICE_FORMAT_JSON = 0,
    POLYVOICE_FORMAT_RTTM = 1,
    POLYVOICE_FORMAT_SRT = 2,
    POLYVOICE_FORMAT_VTT = 3,
    POLYVOICE_FORMAT_TXT = 4
} polyvoice_format_t;

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
 * Run diarization and return the result rendered in the requested format
 * (polyvoice_format_t). Same contract as polyvoice_pipeline_run otherwise.
 * RTTM output uses the fixed file id "audio". Unknown formats return
 * POLYVOICE_ERR_INVALID_ARG. Free *out_str with polyvoice_free_string.
 */
int polyvoice_pipeline_run_format(PolyvoicePipeline* pipeline,
                                  const float* samples,
                                  size_t n_samples,
                                  uint32_t sample_rate,
                                  polyvoice_format_t format,
                                  char** out_str,
                                  size_t* out_str_len);

/**
 * Destroy a pipeline. Must be called exactly once per handle.
 * Must not be called concurrently with run on the same handle.
 */
void polyvoice_pipeline_destroy(PolyvoicePipeline* pipeline);

/** Free a string returned by polyvoice_pipeline_run / polyvoice_pipeline_run_format. */
void polyvoice_free_string(char* p, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* POLYVOICE_H */
