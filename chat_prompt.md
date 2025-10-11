# Chat Prompt

This file describes conventions for assistants working in this repo.

- ONNX export: BaseRunner writes a single file at `output/<env>_<task>.onnx`. Intermediates overwrite unless `--export_on_finish` is set; final uses the same filename.
- Checkpoints: `output/checkpoints/`
- TensorBoard: `output/runs/`
- Pycache: `output/__pycache__/` (set in runner)
- No `.onnx` files should remain in `output/checkpoints/`.
- Keep diffs minimal and focused; avoid refactors unless requested.
- Use valid JSON (no comments). Put notes in a `notes` array if needed.

