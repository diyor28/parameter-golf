This folder is a working copy of the baseline trainer for iterative baseline-focused work.

Contents:
- `train_gpt.py`: copied from the baseline record and intended to be edited here
- `runpod/`: Python launcher, remote training payloads, and configs for this record

Recommended flow:
1. Edit `train_gpt.py` in this folder.
2. `cd runpod`
3. Launch a run with `just run`.
4. Check progress with `just status`.

See `runpod/README.md` for the exact commands.
