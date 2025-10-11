# Copilot Instructions for Open_Duck_Playground

This file defines repo-wide guidance for AI assistants and is the single source of truth for assistant behavior in this project.

---

- ONNX export: BaseRunner writes a single file at `output/<env>_<task>.onnx`. Intermediates overwrite unless `--export_on_finish` is set; final uses the same filename.
- Checkpoints: `output/checkpoints/`
- TensorBoard: `output/runs/`
- Pycache: `output/__pycache__/` (set in runner)
- The `open_min_duck_v2` folder is the reference; the project trains a new model in `playground/wildrobot_dev`.

High-level rules (must follow)
- ALWAYS show a unified diff or clear patch with file paths before applying anything.
- NEVER run `git commit` or `git push` or any terminal commands that modify the repo automatically. Suggest exact commands instead (unless the user explicitly asks you to run them).
- Provide a short one-line summary + 3–5 bullet rationale for any change.
- Keep code snippets minimal and idiomatic.
- Keep diffs minimal and focused; prefer concise, simple design.
- The code in open_duck_mini_v2, wild_robot, wildrobot_dev should not direct reference across folders. the shared logic can be moved to common folder, otherwise, have own implementation in each folder.
