# Runpod Workflow

This workflow is now centered around one happy path:

```bash
cd /Users/diyorkhaydarov/Projects/toys/parameter-golf/records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod
just run
```

That command now does all of the following under the hood:
- reuses a matching running pod if one already exists
- otherwise creates a fresh pod
- waits for real SSH readiness
- syncs the local repo snapshot via `rsync`
- runs remote bootstrap only if the pod is not already bootstrapped
- launches training
- stops the pod automatically when the run finishes

## Everyday commands

Run the normal 2x5090 experiment:

```bash
just run
```

Run the smoke check:

```bash
just smoke
```

Run the 8xH100 record config:

```bash
just record
```

Give a run an explicit name:

```bash
just run report_batch_01
```

Pass one-off env overrides without editing checked-in configs:

```bash
just run report_batch_02 "TRAIN_BATCH_TOKENS=98304 MUON_WEIGHT_DECAY=0.02"
```

Argument convention is positional:
- first arg: optional run name
- second arg: optional extra env assignments

If you do not pass a run name, the launcher now generates one from the live config.
The default naming scheme includes:
- run purpose (`experiment`, `smoke`, `record`)
- hardware (`2x5090`, `8xh100`)
- tokenizer/data variant
- model shape
- batch shape
- MTP setting
- UTC timestamp

Example generated name:

```text
experiment_2x5090_sp1024_l9d512h8kv4-relu2_tbt131072-ga1_mtp0_0320T110500Z
```

## Diagnostics

These are the only `just` diagnostics we keep surfaced:

```bash
just status
just ssh
just stop
just delete
just summarize
```

`just status` shows:
- the current pod id, name, status, and SSH endpoint
- the remote branch
- the latest run dir
- active training processes
- the latest train log tail
- the current W&B run URL when available

## Design choices

- `rsync` is the default sync path. We no longer rely on `scp -r` as the normal workflow.
- `rsync` runs with `--no-owner --no-group` to avoid the permission noise we hit on Runpod volumes.
- The local repo is the source of truth. We do not manage a mutable remote git checkout anymore.
- We sync the repo snapshot, excluding bulky local-only paths like datasets, logs, and `.venv-runpod`.
- We only reuse already-running pods by default. Stopped pods are not resumed automatically.
- The pod configs still default to the official Runpod PyTorch 2.8.0 template `runpod-torch-v280`.
- The experiment pod uses `2x NVIDIA GeForce RTX 5090` on `SECURE` cloud only.
- New launches fail fast if secure capacity is unavailable.

## Current experiment defaults

2x5090 experiment:
- `GRAD_ACCUM_STEPS=1`
- `GPU_COUNT=2`
- `TRAIN_BATCH_TOKENS=131072`
- `VAL_BATCH_SIZE=131072`
- `MUON_BACKEND_STEPS=4`
- `MUON_WEIGHT_DECAY=0.01`
- `EMA_BETA=0.999`
- `EMA_START_PCT=0.8`

Additional supported tuning knobs:
- `MLP_KIND=relu2|swiglu`
- `MLP_HIDDEN_DIM=...`
- `MTP_DEPTH=...`
- `MTP_WEIGHT=...`
- `MUON_UPDATE_SCALE=...`
- `INT8_CLIP_PERCENTILE=...`
- `INT8_KEEP_FLOAT_MAX_NUMEL=...`

Example MTP run:

```bash
just run mtp_depth1 "MTP_DEPTH=1 MTP_WEIGHT=0.3"
```

## Secrets

W&B auth is read from:

`records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/secrets.env`

That file is local-only and gitignored.

## Local analysis

The local summary helper uses Miniconda Python:

```bash
just summarize
```
