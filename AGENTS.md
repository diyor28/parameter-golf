# AGENTS.md

## Purpose

This repository is being used for OpenAI's Parameter Golf challenge work.

The active engineering focus is the working record at:

`records/track_10min_16mb/2026-03-19_WorkingBaseline`

That folder contains:
- a copied baseline `train_gpt.py` that we are iterating on directly
- a Runpod-based experiment workflow for cheap single-GPU iteration and eventual 8xH100 record attempts

## Current problem

We are trying to establish a clean, repeatable baseline on a single RTX 5090 before making architecture or training changes.

That means:
- the remote workflow must be fast to spin up repeatedly
- the pod lifecycle must be reliable enough that setup overhead does not dominate experimentation
- run artifacts and monitoring must be easy to inspect

We already chose:
- Runpod as the remote GPU provider
- the official PyTorch 2.8.0 Runpod template
- Weights & Biases for experiment monitoring

## Workflow decisions

The only Runpod workflow we care about is the record-local one under:

`records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod`

Important conventions:
- prefer the new `pod_*` local scripts and `remote_*` remote scripts
- breaking workflow changes are acceptable; there is no team compatibility requirement
- do not preserve obsolete compatibility wrappers just to avoid churn
- prefer a single obvious happy path over multiple overlapping ways to do the same thing
- the local run launcher should stop the pod automatically when training finishes
- the default workflow should reuse an existing matching running pod when possible, otherwise create a fresh pod
- the default experiment launch path should use `rsync`, not `scp`
- the local repo snapshot is the source of truth; do not manage the remote workspace with `git pull` / `git checkout`
- use `SECURE` cloud only for new launches; do not fall back to `COMMUNITY`

The intended commands are now:
- `just run` for the normal 2x5090 experiment run
- `just smoke` for a tiny health check
- `just record` for the 8xH100 path
- `just status`, `just ssh`, `just stop`, `just delete`, and `just summarize` for diagnostics

The launcher should do the whole experiment setup under the hood:
- reuse a running matching pod when possible
- create a new pod when no running match exists
- sync the repo snapshot from local via `rsync`
- run bootstrap only when the pod is not already bootstrapped
- start training
- stop the pod automatically at the end

Run naming convention:
- if the user does not explicitly pass a run name, generate one from the active config
- default names should reflect purpose, hardware, tokenizer/data variant, model shape, batch shape, and key experiment toggles like MTP
- avoid generic names like `baseline`, `experiment`, or `smoke` without configuration detail

If `just` is installed, the preferred shortcuts are in:

`records/track_10min_16mb/2026-03-19_WorkingBaseline/runpod/justfile`

## Reliability notes from actual use

These came from real usage and should inform future changes:
- 5090 experimentation should use `SECURE` cloud only
- if secure capacity is unavailable, fail clearly instead of silently or manually falling back to `COMMUNITY`
- "pod exists" is not enough; SSH readiness should be verified by actually connecting
- setup cost matters a lot, so cached venvs and cached dataset shards are important
- local/remote script naming needs to stay explicit to avoid running remote payloads locally by accident
- `rsync` is preferred over `scp -r` for repeated syncs

## Monitoring and artifacts

Primary monitoring:
- Weights & Biases

## Local Python tooling

For local analysis scripts, always use the Miniconda Python at:

`~/miniconda3/bin/python`

Reason:
- the macOS system/Xcode Python and ad hoc local `pip` installs were flaky in practice
- this repo is only being used by us on this machine, so local tooling should assume this Miniconda install exists
- Miniconda already has `pandas` available locally and is the base for future analysis tooling

Practical rule:
- default to `~/miniconda3/bin/python` for local metrics inspection, W&B API pulls, pandas scripts, and one-off analysis commands
- do not use the macOS system/Xcode Python for repo-local analysis
- do not attempt fresh local Python installs into other interpreters for repo-local analysis

Local artifacts per run:
- `logs/record_runs/<RUN_ID>/train.log`
- `logs/record_runs/<RUN_ID>/events.jsonl`
- `logs/record_runs/<RUN_ID>/summary.json`
- `logs/record_runs/<RUN_ID>/console.log`
- `logs/record_runs/<RUN_ID>/run.env`
- `logs/record_runs/<RUN_ID>/git_commit.txt`

## Scope guidance

When changing code for this effort:
- prefer editing the working record copy instead of the root baseline unless there is a strong reason not to
- keep the Runpod workflow centered in the working record folder
- optimize for iteration speed, observability, and reproducibility
- treat the 2x5090 experiment path as the reference point before 8xH100 record attempts
