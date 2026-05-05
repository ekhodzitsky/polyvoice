#ifndef POLYVOICE_H
#define POLYVOICE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a diarizer instance.
typedef struct PolyvoiceDiarizer PolyvoiceDiarizer;

/// A single speaker turn.
typedef struct {
    char* speaker;
    float start;
    float end;
} PolyvoiceTurn;

/// Result of diarization.
typedef struct {
    PolyvoiceTurn* turns;
    size_t num_turns;
} PolyvoiceResult;

/// Create a new diarizer.
/// Returns NULL on failure. Must be freed with `polyvoice_diarizer_free`.
PolyvoiceDiarizer* polyvoice_diarizer_new(float threshold, int max_speakers);

/// Run diarization on mono f32 samples at 16 kHz.
/// Returns NULL on error. Must be freed with `polyvoice_result_free`.
PolyvoiceResult* polyvoice_diarizer_run(PolyvoiceDiarizer* diarizer,
                                        const float* samples,
                                        size_t sample_count);

/// Free a diarizer instance.
/// Passing NULL is a no-op. Double-free is undefined behaviour.
void polyvoice_diarizer_free(PolyvoiceDiarizer* diarizer);

/// Free a result returned by `polyvoice_diarizer_run`.
/// Passing NULL is a no-op. Double-free is undefined behaviour.
void polyvoice_result_free(PolyvoiceResult* result);

/// Return the library version as a static C string.
const char* polyvoice_version(void);

#ifdef __cplusplus
}
#endif

#endif /* POLYVOICE_H */
