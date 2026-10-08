# Analysis contract v1

This directory is the source of truth shared by the Python Server pipeline, Swift Local pipeline, backend, and Flutter app.

Rules:

- `schema_version` is `1.0.0` for every v1 document.
- One `AnalysisRequest` may target Server, Local, or both. Each target creates a separate `AnalysisRun`.
- Camera indices are zero-based, unique, contiguous, and sorted ascending.
- Video hashes identify exact input bytes. Server and Local runs are comparable only when their request and video hashes match.
- Pixel coordinates use a top-left origin, +x right, +y down, and original oriented video pixels.
- Bounding boxes use `[x1, y1, x2, y2]`.
- Artifact paths are relative to the result bundle and must not contain `..`.
- JSON numeric values must be finite. Unknown optional measurements are `null`, never NaN or infinity.
- A Comparison Report references two Result Manifests; it does not merge or overwrite them.

Run contract tests with:

```bash
pytest -q tests/contracts
```
