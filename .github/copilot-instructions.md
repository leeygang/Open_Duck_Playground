# Copilot Instructions for Open_Duck_Playground

This file enables repo-wide guidance for AI assistants. It mirrors the conventions in `chat_prompt.md` so they are picked up automatically by tools that read `.github/copilot-instructions.md`.

Source of truth: `chat_prompt.md` at repo root.

---

- ONNX export: BaseRunner writes a single file at `output/<env>_<task>.onnx`. Intermediates overwrite unless `--export_on_finish` is set; final uses the same filename.
- Checkpoints: `output/checkpoints/`
- TensorBoard: `output/runs/`
- Pycache: `output/__pycache__/` (set in runner)
- The `open_min_duck_v2` folder is the reference; the project trains a new model in `playground/wildrobot_dev`.

High-level rules (must follow)
- ALWAYS show a unified diff or clear patch with file paths before applying anything.
- NEVER run `git commit` or `git push` or any terminal commands that modify the repo automatically. Suggest exact commands instead.
- Provide a short one-line summary + 3–5 bullet rationale for any change.
- Keep code snippets minimal and idiomatic.
- Keep diffs minimal and focused; prefer concise, simple design.
