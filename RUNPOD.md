# Runpod Workflow

This setup is aimed at two things:
- repeatable runs, by checking the launch config into the repo
- cheap iteration, by making the 1 GPU smoke path the default first step

## Recommended pod strategy

Use two modes:
- `1x` lower-cost GPU or spot pod for smoke tests
- `8xH100 SXM` only when you are ready to measure a leaderboard-style run

Cost control tips:
- Start with `configs/runpod/smoke_1gpu.env`
- Download only `1` train shard for smoke tests
- Keep `VAL_LOSS_EVERY=0` while verifying the loop
- Reuse a volume or keep the pod alive long enough to avoid redownloading data

## First-time pod setup

The official Runpod template in the repo README already has the Python dependencies installed. On the pod:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
chmod +x scripts/runpod_train.sh
```

If you are not using the official template, install the requirements first:

```bash
pip install -r requirements.txt
```

## Cheap smoke run

This validates that `torchrun`, data download, and the training loop work on your pod:

```bash
./scripts/runpod_train.sh ./configs/runpod/smoke_1gpu.env
```

Artifacts land in `logs/runpod/<RUN_ID>/`:
- `train.log`
- `run.env`
- `git_commit.txt`

That makes reruns auditable and easy to compare.

## 8 GPU baseline run

Once the smoke run is clean, launch the larger run:

```bash
./scripts/runpod_train.sh ./configs/runpod/baseline_8gpu.env
```

## Making runs reproducible

Copy one of the checked-in config files and edit only the variables you care about:

```bash
cp configs/runpod/smoke_1gpu.env configs/runpod/my_experiment.env
./scripts/runpod_train.sh ./configs/runpod/my_experiment.env
```

Because the wrapper saves both the effective environment and the Git commit, each run has:
- the exact hyperparameter environment
- the exact code revision
- the exact training log

## Useful overrides

You can keep the config file static and override a few fields from the shell:

```bash
RUN_ID=test_seq2048 TRAIN_SEQ_LEN=2048 ./scripts/runpod_train.sh ./configs/runpod/smoke_1gpu.env
```

## Notes

- `DOWNLOAD_TRAIN_SHARDS` controls how much training data is fetched before the run.
- `DATA_PATH` and `TOKENIZER_PATH` are filled in automatically for the selected `DOWNLOAD_VARIANT`.
- Set `SKIP_DATA_DOWNLOAD=1` if the dataset is already present on disk.
