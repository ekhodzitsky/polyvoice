#include "polyvoice.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    FILE *file = fopen(argv[2], "rb");
    if (!file) return 3;
    if (fseek(file, 0, SEEK_END) != 0) return 4;
    long bytes = ftell(file);
    if (bytes < 0 || bytes % sizeof(float) || bytes > (16000L * 3600 + 1) * 4) return 5;
    rewind(file);
    float *samples = malloc(bytes ? (size_t)bytes : sizeof(float));
    if (!samples) return 6;
    size_t count = (size_t)bytes / sizeof(float);
    if (fread(samples, sizeof(float), count, file) != count) return 7;
    fclose(file);
    PolyvoicePipeline *pipeline = NULL;
    int rc = polyvoice_pipeline_create(POLYVOICE_PROFILE_BALANCED, argv[1], &pipeline);
    if (rc != POLYVOICE_OK || !pipeline) return 8;
    char *json = NULL;
    size_t length = 0;
    int invalid_rate = strcmp(argv[3], "invalid-rate") == 0;
    int too_long = strcmp(argv[3], "too-long") == 0;
    rc = polyvoice_pipeline_run(pipeline, samples, count, invalid_rate ? 8000 : 16000, &json, &length);
    free(samples);
    int too_short = strcmp(argv[3], "empty") == 0 || strcmp(argv[3], "short") == 0;
    if (invalid_rate || too_long || too_short) {
        int expected = invalid_rate ? POLYVOICE_ERR_INVALID_ARG : (too_long ? POLYVOICE_ERR_AUDIO_TOO_LONG : POLYVOICE_ERR_INFERENCE);
        if (rc != expected || json != NULL || length != 0) return 9;
        printf("{\"error\":\"%s\"}", argv[3]);
        polyvoice_pipeline_destroy(pipeline);
        return 0;
    }
    if (rc != POLYVOICE_OK || !json || !length) return 10;
    if (fwrite(json, 1, length, stdout) != length) return 11;
    polyvoice_free_string(json, length);
    polyvoice_pipeline_destroy(pipeline);
    return 0;
}
