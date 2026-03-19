This folder is a working copy of the baseline trainer for iterative experiments.

Contents:
- `train_gpt.py`: copied from the baseline record and intended to be edited here
- `runpod/`: local Runpod CLI helpers plus remote training wrappers for this record

Recommended flow:
1. Edit `train_gpt.py` in this folder.
2. Push your branch or otherwise make this record available on the remote pod.
3. Create a cheap smoke-test pod with `runpod/create_pod.sh`.
4. SSH into the pod and run `runpod/train_remote.sh` from the cloned repo.

See `runpod/README.md` for the exact commands.
