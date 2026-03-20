This folder is a working copy of the baseline trainer for iterative baseline-focused work.

Contents:
- `train_gpt.py`: copied from the baseline record and intended to be edited here
- `runpod/`: local pod management scripts plus remote training payloads for this record

Recommended flow:
1. Edit `train_gpt.py` in this folder.
2. Prepare a fresh dual-5090 pod with `runpod/pod_prepare.sh`.
3. Launch a run with `runpod/pod_run.sh` or `runpod/justfile`.
4. Check progress with `runpod/pod_status.sh`.

See `runpod/README.md` for the exact commands.
