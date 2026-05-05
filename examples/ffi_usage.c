/* C FFI usage example.
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
    printf("polyvoice version: %s\n", polyvoice_version());

    PolyvoiceDiarizer *d = polyvoice_diarizer_new(0.5f, 64);
    if (!d) {
        fprintf(stderr, "failed to create diarizer\n");
        return 1;
    }

    /* 2 seconds of silence @ 16 kHz */
    size_t sample_count = 16000 * 2;
    float *samples = calloc(sample_count, sizeof(float));
    if (!samples) {
        polyvoice_diarizer_free(d);
        return 1;
    }

    PolyvoiceResult *r = polyvoice_diarizer_run(d, samples, sample_count);
    free(samples);

    if (r) {
        printf("Detected %zu turn(s):\n", r->num_turns);
        for (size_t i = 0; i < r->num_turns; i++) {
            printf("  %s: %.2f - %.2f\n",
                   r->turns[i].speaker,
                   r->turns[i].start,
                   r->turns[i].end);
        }
        polyvoice_result_free(r);
    } else {
        printf("No turns detected (audio too short or error).\n");
    }

    polyvoice_diarizer_free(d);
    return 0;
}
