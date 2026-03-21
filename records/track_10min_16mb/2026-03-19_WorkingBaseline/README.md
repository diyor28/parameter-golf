This folder is a working copy of the baseline trainer for iterative record-track work.

Contents:
- `train_gpt.py`: the active working trainer, now carrying the stronger record-style stack
  (11L/512/MLP3x, SmearGate, BigramHash, XSA, partial RoPE, LN scale, late QAT,
  mixed int5/int6 + zstd, sliding-window eval, and causal TTT)
- `runpod/`: Python launcher, remote training payloads, and configs for this record

Recommended flow:
1. Edit `train_gpt.py` in this folder.
2. `cd runpod`
3. Launch a run with `just run`.
4. Check progress with `just status`.

See `runpod/README.md` for the exact commands.
