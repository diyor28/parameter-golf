# Prebuilt Runpod Image

This image is meant to eliminate the most expensive first-run setup step on fresh pods:

- creating a virtualenv is still cheap
- repeated `pip install` is not

The image is based on the official Runpod PyTorch 2.8.0 image and bakes in the Python
dependencies used by the Parameter Golf workflow.

## Build and publish

Build from this directory:

```bash
docker build -t <your-registry>/parameter-golf-runpod:torch280 .
docker push <your-registry>/parameter-golf-runpod:torch280
```

Then point the pod config at the image:

```bash
RUNPOD_IMAGE=<your-registry>/parameter-golf-runpod:torch280
SKIP_PIP_INSTALL=1
```

That causes the launcher to create pods with `--image` instead of `--template-id`, and it
causes `remote_bootstrap.sh` to skip the expensive package installation step.
