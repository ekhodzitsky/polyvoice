#include "polyvoice.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *file = fopen(argv[2], "rb");
    if (!file) return 3;
    if (fseek(file, 0, SEEK_END) != 0) return 4;
    long bytes = ftell(file);
    if (bytes <= 0 || bytes % sizeof(float) || bytes > 16000 * 60 * 4) return 5;
    rewind(file);
    float *samples = malloc((size_t)bytes);
    if (!samples) return 6;
    size_t count = (size_t)bytes / sizeof(float);
    if (fread(samples, sizeof(float), count, file) != count) return 7;
    fclose(file);
    PolyvoicePipeline *pipeline = NULL;
    int rc = polyvoice_pipeline_create(POLYVOICE_PROFILE_BALANCED, argv[1], &pipeline);
    if (rc != POLYVOICE_OK || !pipeline) return 8;
    char *json = NULL;
    size_t length = 0;
    rc = polyvoice_pipeline_run(pipeline, samples, count, 1, &json, &length);
    if (rc != POLYVOICE_ERR_INVALID_ARG || json != NULL) return 9;
    rc = polyvoice_pipeline_run(pipeline, samples, count, 16000, &json, &length);
    free(samples);
    if (rc != POLYVOICE_OK || !json || !length) return 10;
    if (fwrite(json, 1, length, stdout) != length) return 11;
    polyvoice_free_string(json, length);
    polyvoice_pipeline_destroy(pipeline);
    return 0;
}
