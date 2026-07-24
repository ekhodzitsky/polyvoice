# TODO — src/models

## Current

## Next

- [ ] Add retry with exponential backoff for downloads.
- [ ] Add progress callback API.
- [ ] Wire AdapterRegistry into pipeline_v2 builder (string-config path).
- [ ] After injecting metadata_props into shipped ONNX: recompute hashes + re-sign.

## Known Gaps

- Shipped ONNX binaries in this tree are often absent (only `.minisig` sidecars);
  schema-v2 manifest fields are the fallback until injection + re-sign lands.

## Deferred

- [ ] Support for partial/resumable downloads.
- [ ] Release-key re-signing of models after metadata injection (release workflow).
