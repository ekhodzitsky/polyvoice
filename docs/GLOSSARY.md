# GLOSSARY.md — Vocabulary for polyvoice

## Core Terms

**Invariant** — A condition that is always true at a given point in the code. Example: `SpeakerCluster.centroids.len() <= config.max_speakers`.

**Precondition** — A requirement that must hold before a function is called. Example: `!samples.is_empty()` for `extract()`.

**Postcondition** — A guarantee that holds after a function returns. Example: `embedding.len() == EMBEDDING_DIM`.

**Hoare Triple** — Notation `{P} C {Q}` where P is precondition, C is code, Q is postcondition.

**Newtype** — A wrapper struct around a primitive that adds semantic meaning and invariants.

**Typestate** — Encoding state machine states as different types, making illegal transitions compile-time errors.

**Embedding** — A dense vector representation of a speaker's voice characteristics (typically 128–1024 dimensions).

**Centroid** — The mean vector of all embeddings assigned to a speaker cluster.

**Cosine Similarity** — Dot product of two normalized vectors; measures directional similarity in range [-1, 1].

**Diarization Error Rate (DER)** — Metric: fraction of time incorrectly attributed to speakers.

## Rust-Specific

**PhantomData** — Zero-sized type used to mark ownership or variance of generic parameters.

**SAFETY Comment** — Required documentation block before every `unsafe` block explaining why the code is sound.
