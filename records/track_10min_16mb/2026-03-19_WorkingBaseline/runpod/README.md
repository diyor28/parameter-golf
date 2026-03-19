# Runpod CLI Workflow

This workflow uses `runpodctl` locally for pod lifecycle and uses the record-local
`train_gpt.py` on the remote machine.

The pod configs default to the official Runpod PyTorch 2.8.0 template id
`runpod-torch-v280`, which your local `runpodctl template search "pytorch 2.8"`
returned as `Runpod Pytorch 2.8.0`.

Experiment tracking uses Weights & Biases when `WANDB_ENABLE=1`.

## Files

- `pod_experiment_1x5090.env`: local pod settings for experimentation on a single RTX 5090
- `pod_record_8xh100.env`: local pod settings for a record attempt on 8xH100
- `train_smoke_1x5090.env`: tiny checked-in smoke config for quick health checks
- `train_experiment_1x5090.env`: remote training settings for the 1x5090 experiment loop
- `train_record_8xh100.env`: remote training settings for the 8xH100 record run
- `create_pod.sh`: create a pod with `runpodctl`
- `start_pod.sh`: start a stopped pod
- `stop_pod.sh`: stop a pod to stop billing compute
- `delete_pod.sh`: terminate a pod
- `sync_record.sh`: copy just this working record to the current pod
- `bootstrap_pod.sh`: one-time pod bootstrap for venv, Python deps, and cached data
- `train_remote.sh`: run this record's `train_gpt.py` on the pod
- `summarize_runs.py`: compare completed runs locally from `logs/record_runs`
- `justfile`: short aliases for the common pod, sync, bootstrap, smoke, and run commands

If you have [`just`](https://github.com/casey/just) installed, run `just --list` in this
folder to see the shortcut commands.

## Recommended Loop

For repeated experiments, keep the same 1x5090 pod around and reuse it:

```bash
cd /Users/diyorkhaydarov/Projects/toys/parameter-golf/records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod
./create_pod.sh ./pod_experiment_1x5090.env
./sync_record.sh
```

SSH into the pod once, then run the one-time bootstrap:

```bash
bash records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/bootstrap_pod.sh \
  records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_experiment_1x5090.env
```

After that, `train_remote.sh` auto-detects the cached venv and dataset and skips setup work.

## Experiment run: 1x RTX 5090

On your local machine:

```bash
cd /Users/diyorkhaydarov/Projects/toys/parameter-golf/records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod
./create_pod.sh ./pod_experiment_1x5090.env
```

That prints the pod id and an SSH hint once the pod is ready.

Then SSH into the pod, clone your branch, and from the repo root run:

```bash
bash records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_remote.sh \
  records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_experiment_1x5090.env
```

For a very short health check:

```bash
bash records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_remote.sh \
  records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_smoke_1x5090.env
```

## Record run: 8xH100

Local machine:

```bash
./create_pod.sh ./pod_record_8xh100.env
```

Remote pod:

```bash
bash records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_remote.sh \
  records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_record_8xh100.env
```

## W&B setup

This workflow reads `WANDB_API_KEY` locally from:

`records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/secrets.env`

That file is ignored by git. The key is used at train-launch time and is no longer
injected into pod creation metadata.

If you prefer manual auth on the pod, the simplest path is:

```bash
wandb login
```

If `WANDB_ENABLE=1`, `train_remote.sh` will install `wandb` on the pod automatically
if it is missing.

The current configs already enable W&B:
- `train_experiment_1x5090.env` logs to project `parameter-golf`, group `experiments`
- `train_record_8xh100.env` logs to project `parameter-golf`, group `record-attempts`

You can also set:
- `WANDB_ENTITY`
- `WANDB_NOTES`
- `WANDB_TAGS`
- `WANDB_RUN_NAME`

## Shell Overrides

The checked-in `.env` files now use shell-friendly defaults, so overrides work cleanly:

```bash
RUN_ID=my_test ITERATIONS=20 TRAIN_BATCH_TOKENS=16384 \
bash records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_remote.sh \
  records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/train_smoke_1x5090.env
```

## Cost control

- Use `pod_experiment_1x5090.env` first. It defaults to `1x RTX 5090` on community cloud.
- Use `train_smoke_1x5090.env` first for very short checks.
- Reuse the same pod with `./stop_pod.sh` and `./start_pod.sh` instead of deleting and recreating it.
- Stop or delete pods as soon as a run ends:

```bash
./stop_pod.sh <pod-id>
./delete_pod.sh <pod-id>
```

## Reproducibility

`train_remote.sh` writes per-run artifacts under `logs/record_runs/<RUN_ID>/`:
- `train.log`
- `events.jsonl`
- `summary.json`
- `run.env`
- `git_commit.txt`
- `console.log`

That gives you the code revision, effective environment, and full log for each run.

## Monitoring approach

Weights & Biases is the primary monitoring surface for this workflow.

The trainer also keeps local artifacts in case you want to inspect or compare runs offline:
- `train.log`: human-readable trainer log
- `events.jsonl`: machine-readable event stream for train, val, artifact, and final eval events
- `summary.json`: latest rollup for quick comparison across runs

After you pull `logs/record_runs` back to your local machine, you can still compare runs with:

```bash
python3 records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/summarize_runs.py \
  logs/record_runs
```
