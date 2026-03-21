# Runpod Workflow

This workflow is now centered around one Python entrypoint:

```bash
cd /Users/diyorkhaydarov/Projects/toys/parameter-golf/records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod
just run
```

`just` now calls [launcher.py](/Users/diyorkhaydarov/Projects/toys/parameter-golf/records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/launcher.py), which does all of the following under the hood:
- reuses a matching running pod if one already exists
- otherwise creates a fresh pod
- waits for real SSH readiness
- syncs only the required local snapshot via `rsync`
- runs remote bootstrap only if the pod is not already bootstrapped
- launches training
- stops the pod automatically when the run finishes

Auto-stop is layered on purpose:
- the remote training script now schedules a `podStop` call itself via the Runpod API
- the local launcher still issues a stop request in its `finally` block as a backup

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
- UTC timestamp

Example generated name:

```text
experiment_2x5090_sp1024_l9d512h8kv4-relu2_tbt65536_0320T110500Z
```

## Diagnostics

These are the only `just` diagnostics we keep surfaced:

```bash
just status
just ssh
just stop
just stop qc6w4g4h2kzn7m
just delete
just summarize
```

`just status` shows:
- the current pod id, name, status, and SSH endpoint
- the latest run dir
- active training processes
- the latest train log tail
- the current W&B run URL when available

## Design choices

- `rsync` is the default sync path. We no longer rely on `scp -r` as the normal workflow.
- `rsync` runs with `--no-owner --no-group` to avoid the permission noise we hit on Runpod volumes.
- The local repo is the source of truth. We do not manage a mutable remote git checkout.
- We sync only the paths the run actually needs: `data/` and the working record folder.
- We only reuse already-running pods by default. Stopped pods are not resumed automatically.
- The pod configs still default to the official Runpod PyTorch 2.8.0 template `runpod-torch-v280`.
- The experiment pod uses `2x NVIDIA GeForce RTX 5090` on `SECURE` cloud only.
- New launches fail fast if secure capacity is unavailable.
- There is now a first-class path to a prebuilt custom image via `RUNPOD_IMAGE`, which avoids repeated `pip install` on fresh pods.

## Current experiment defaults

2x5090 experiment:
- `GPU_COUNT=2`
- `TRAIN_BATCH_TOKENS=65536`
- `VAL_BATCH_SIZE=65536`
- `MUON_BACKEND_STEPS=5`
- `SKIP_PIP_INSTALL=0`

Additional supported tuning knobs:
- `TRAIN_BATCH_TOKENS=...`
- `VAL_BATCH_SIZE=...`
- `ROUNDTRIP_EVAL_EVERY=...`

## Secrets

W&B auth is read from:

`records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/secrets.env`

That file is local-only and gitignored.

## Prebuilt image

The long-term fast path is a custom Runpod image that already has the Python
dependencies installed.

Files:
- [image/Dockerfile](/Users/diyorkhaydarov/Projects/toys/parameter-golf/records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/image/Dockerfile)
- [image/requirements.txt](/Users/diyorkhaydarov/Projects/toys/parameter-golf/records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/image/requirements.txt)
- [image/README.md](/Users/diyorkhaydarov/Projects/toys/parameter-golf/records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/image/README.md)

Once that image is published, set:

```bash
RUNPOD_IMAGE=<your-image>
SKIP_PIP_INSTALL=1
```

That removes the slowest first-run setup step on new pods.

## Local analysis

The local summary helper uses Miniconda Python:

```bash
just summarize
```
