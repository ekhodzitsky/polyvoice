/* C FFI usage example for polyvoice ABI v3.
 *
 * Build the shared library first:
 *     cargo build --features ffi
 *
 * Compile this example:
 *     cc -I include examples/ffi_usage.c -L target/debug -lpolyvoice -o ffi_usage
 *
 * Run:
 *     LD_LIBRARY_PATH=target/debug ./ffi_usage
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "polyvoice.h"

int main(void) {
    PolyvoicePipeline* handle = NULL;

    int rc = polyvoice_pipeline_create(
        POLYVOICE_PROFILE_BALANCED,
        NULL,  /* use default model cache */
        &handle
    );
    if (rc != POLYVOICE_OK || !handle) {
        fprintf(stderr, "failed to create pipeline (status=%d)\n", rc);
        return 1;
    }

    /* 2 seconds of silence @ 16 kHz */
    size_t sample_count = 16000 * 2;
    float* samples = calloc(sample_count, sizeof(float));
    if (!samples) {
        polyvoice_pipeline_destroy(handle);
        return 1;
    }

    char* json = NULL;
    size_t json_len = 0;
    rc = polyvoice_pipeline_run(
        handle,
        samples,
        sample_count,
        16000,  /* sample rate */
        &json,
        &json_len
    );
    free(samples);

    if (rc == POLYVOICE_OK && json) {
        printf("Result (first 256 chars): %.256s\n", json);
        polyvoice_free_string(json, json_len);
    } else {
        fprintf(stderr, "pipeline run failed (status=%d)\n", rc);
    }

    polyvoice_pipeline_destroy(handle);
    return 0;
}
