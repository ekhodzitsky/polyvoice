# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of `polyvoice`.
- `EmbeddingExtractor` trait with `DummyExtractor` (tests) and `OnnxEmbeddingExtractor` (ONNX Runtime, pooled sessions).
- `SpeakerCluster` with online incremental centroid updates and cosine-similarity threshold.
- `OnlineDiarizer` for real-time streaming with sliding windows.
- `OfflineDiarizer` for file-based diarization with segment merging and gap filling.
- `VoiceActivityDetector` trait with `EnergyVad` reference implementation.
- Overlap detection (`detect_overlaps`) from fine-grained segment lists.
- Word-level speaker alignment (`OnlineDiarizer::align_words`).
- Comprehensive unit and integration tests.
