#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RECORD_DIR = SCRIPT_DIR.parent
REPO_ROOT_LOCAL = SCRIPT_DIR.parents[3]
STATE_ROOT = SCRIPT_DIR / "state"
CURRENT_POD_ID_PATH = STATE_ROOT / "current_pod_id"
CURRENT_POD_JSON_PATH = STATE_ROOT / "current_pod.json"
REPO_ROOT_REMOTE = "/workspace/parameter-golf"
RECORD_ROOT_REMOTE = "records/track_10min_16mb/2026-03-19_WorkingBaseline"
MINICONDA_PYTHON = Path.home() / "miniconda3" / "bin" / "python"
DEFAULT_POD_CONFIG = SCRIPT_DIR / "pod_experiment_2x5090.env"
DEFAULT_TRAIN_CONFIG = SCRIPT_DIR / "train_experiment_2x5090.env"
DEFAULT_RECORD_POD_CONFIG = SCRIPT_DIR / "pod_record_8xh100.env"
DEFAULT_RECORD_TRAIN_CONFIG = SCRIPT_DIR / "train_record_8xh100.env"
DEFAULT_SMOKE_TRAIN_CONFIG = SCRIPT_DIR / "train_smoke_2x5090.env"

SHELL_DEFAULT_RE = re.compile(r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\:\-(?P<default>.*)\}$")


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def format_completed_error(exc: subprocess.CalledProcessError) -> str:
    parts = [f"Command failed: {' '.join(exc.cmd)}"]
    if exc.stdout:
        parts.append(exc.stdout.strip())
    if exc.stderr:
        parts.append(exc.stderr.strip())
    return "\n".join(part for part in parts if part)


def stream(args: list[str], *, cwd: Path | None = None) -> int:
    process = subprocess.Popen(args, cwd=str(cwd) if cwd else None)
    return process.wait()


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        match = SHELL_DEFAULT_RE.match(value)
        if match:
            env_name = match.group("name")
            default = match.group("default").strip().strip("'").strip('"')
            values[key] = os.environ.get(env_name, default)
        else:
            values[key] = os.environ.get(key, value.strip("'").strip('"'))
    return values


def parse_extra_env(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    result: dict[str, str] = {}
    for item in shlex.split(raw):
        if "=" not in item:
            raise ValueError(f"Invalid env override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def resolve_pod_id(requested_pod_id: str | None = None) -> str:
    if requested_pod_id:
        return requested_pod_id
    if CURRENT_POD_ID_PATH.exists():
        return CURRENT_POD_ID_PATH.read_text(encoding="utf-8").strip()
    raise SystemExit(f"No pod id provided and no current pod recorded in {CURRENT_POD_ID_PATH}")


def save_current_pod_state(pod_id: str, ssh_info: dict[str, Any] | None = None) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    CURRENT_POD_ID_PATH.write_text(f"{pod_id}\n", encoding="utf-8")
    if ssh_info is not None:
        CURRENT_POD_JSON_PATH.write_text(
            json.dumps({"pod_id": pod_id, "ssh_info": ssh_info}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_current_pod_state(pod_id: str | None = None) -> dict[str, Any] | None:
    if not CURRENT_POD_JSON_PATH.exists():
        return None
    try:
        data = json.loads(CURRENT_POD_JSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if pod_id and data.get("pod_id") != pod_id:
        return None
    return data


def clear_current_pod_state(pod_id: str | None = None) -> None:
    if pod_id and CURRENT_POD_ID_PATH.exists():
        current = CURRENT_POD_ID_PATH.read_text(encoding="utf-8").strip()
        if current != pod_id:
            return
    for path in (CURRENT_POD_ID_PATH, CURRENT_POD_JSON_PATH):
        if path.exists():
            path.unlink()


def runpod_json(args: list[str]) -> Any:
    result = run(["runpodctl", *args, "-o", "json"])
    return json.loads(result.stdout)


def pod_get_json(pod_id: str) -> dict[str, Any]:
    data = runpod_json(["pod", "get", pod_id])
    if not isinstance(data, dict):
        raise SystemExit(f"Unexpected pod JSON for {pod_id}")
    return data


def pod_list_json(*, include_all: bool = False, name: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    args = ["pod", "list"]
    if include_all:
        args.append("--all")
    if name:
        args.extend(["--name", name])
    if status:
        args.extend(["--status", status])
    data = runpod_json(args)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def ssh_info_json(pod_id: str) -> dict[str, Any]:
    last_error: Exception | None = None

    cached = load_current_pod_state(pod_id)
    cached_ssh = cached.get("ssh_info") if cached else None
    if isinstance(cached_ssh, dict) and all(
        cached_ssh.get(field) for field in ("ip", "port", "ssh_command")
    ) and isinstance((cached_ssh.get("ssh_key") or {}), dict) and (cached_ssh.get("ssh_key") or {}).get("path"):
        return cached_ssh

    for _ in range(3):
        try:
            data = runpod_json(["ssh", "info", pod_id])
            if not isinstance(data, dict):
                raise SystemExit(f"Unexpected ssh info JSON for {pod_id}")
            if all(data.get(field) for field in ("ip", "port", "ssh_command")):
                save_current_pod_state(pod_id, data)
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2)

    try:
        pod = pod_get_json(pod_id)
        ssh = pod.get("ssh")
        if isinstance(ssh, dict) and all(ssh.get(field) for field in ("ip", "port", "ssh_command")):
            save_current_pod_state(pod_id, ssh)
            return ssh
    except Exception as exc:
        last_error = exc

    if last_error is not None:
        raise last_error
    raise SystemExit(f"Unexpected ssh info JSON for {pod_id}")


def ssh_parts(pod_id: str) -> tuple[str, str, str, str]:
    info = ssh_info_json(pod_id)
    host = info.get("ip")
    port = info.get("port")
    key_path = (info.get("ssh_key") or {}).get("path")
    ssh_command = info.get("ssh_command")
    if not all((host, port, key_path, ssh_command)):
        raise RuntimeError(f"SSH info incomplete for pod {pod_id}")
    return str(host), str(port), str(key_path), str(ssh_command)


def wait_for_ssh_ready(pod_id: str, attempts: int = 60, interval_seconds: int = 10) -> dict[str, Any]:
    for _ in range(attempts):
        try:
            info = ssh_info_json(pod_id)
            host, port, key_path, _ssh_command = ssh_parts(pod_id)
        except Exception:
            time.sleep(interval_seconds)
            continue
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            key_path,
            "-p",
            port,
            f"root@{host}",
            "echo ready",
        ]
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode == 0 and result.stdout.strip() == "ready":
            save_current_pod_state(pod_id, info)
            return info
        time.sleep(interval_seconds)
    raise TimeoutError(f"Timed out waiting for SSH readiness on pod {pod_id}")


def start_pod(pod_id: str) -> bool:
    eprint(f"==> Starting stopped pod {pod_id}")
    try:
        run(["runpodctl", "pod", "start", pod_id])
        return True
    except subprocess.CalledProcessError as exc:
        message = format_completed_error(exc)
        if "not enough free gpus on the host machine" in message.lower():
            eprint("==> Reuse failed because the old host no longer has free GPUs; creating a fresh pod instead")
            return False
        raise SystemExit(message)


def find_reusable_pod(name: str) -> dict[str, Any] | None:
    current_pod_id = CURRENT_POD_ID_PATH.read_text(encoding="utf-8").strip() if CURRENT_POD_ID_PATH.exists() else None
    candidates = pod_list_json(include_all=True, name=name)
    if not candidates:
        return None

    def rank(pod: dict[str, Any]) -> tuple[int, int]:
        pod_id = str(pod.get("id", ""))
        status = str(pod.get("desiredStatus", ""))
        running_rank = 0 if status == "RUNNING" else 1
        current_rank = 0 if current_pod_id and pod_id == current_pod_id else 1
        return (running_rank, current_rank)

    candidates.sort(key=rank)
    return candidates[0]


def create_pod(config: dict[str, str]) -> str:
    cloud_type = config.get("RUNPOD_CLOUD_TYPE", "SECURE").upper()
    if cloud_type != "SECURE" and config.get("ALLOW_RUNPOD_COMMUNITY", "0") != "1":
        raise SystemExit(
            f"Refusing to create a non-SECURE pod ({config.get('RUNPOD_CLOUD_TYPE')}). "
            "This workflow is secure-only unless ALLOW_RUNPOD_COMMUNITY=1 is set explicitly."
        )

    cmd = [
        "runpodctl",
        "pod",
        "create",
        "--name",
        config.get("POD_NAME", "pgolf-working"),
        "--gpu-id",
        config.get("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 5090"),
        "--gpu-count",
        config.get("RUNPOD_GPU_COUNT", "1"),
        "--cloud-type",
        config.get("RUNPOD_CLOUD_TYPE", "SECURE"),
        "--container-disk-in-gb",
        config.get("RUNPOD_CONTAINER_DISK_GB", "30"),
        "--volume-in-gb",
        config.get("RUNPOD_VOLUME_GB", "40"),
        "--volume-mount-path",
        config.get("RUNPOD_VOLUME_MOUNT_PATH", "/workspace"),
        "-o",
        "json",
    ]
    image = config.get("RUNPOD_IMAGE", "").strip()
    template_id = config.get("RUNPOD_TEMPLATE_ID", "").strip()
    if image:
        cmd.extend(["--image", image])
    elif template_id:
        cmd.extend(["--template-id", template_id])
    else:
        raise SystemExit("Pod config must set RUNPOD_IMAGE or RUNPOD_TEMPLATE_ID")

    if config.get("RUNPOD_SSH", "1") == "1":
        cmd.append("--ssh")
    if config.get("RUNPOD_ENABLE_PUBLIC_IP", "1") == "1" and cloud_type != "SECURE":
        cmd.append("--public-ip")
    if ports := config.get("RUNPOD_PORTS", "").strip():
        cmd.extend(["--ports", ports])
    if data_centers := config.get("RUNPOD_DATA_CENTER_IDS", "").strip():
        cmd.extend(["--data-center-ids", data_centers])
    if env_json := config.get("RUNPOD_ENV_JSON", "").strip():
        cmd.extend(["--env", env_json])
    if config.get("RUNPOD_GLOBAL_NETWORKING", "0") == "1":
        cmd.append("--global-networking")

    eprint(f"==> Creating pod {config.get('POD_NAME', 'pgolf-working')}")
    try:
        result = run(cmd)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(format_completed_error(exc))
    data = json.loads(result.stdout)
    if isinstance(data, list):
        data = data[0] if data else {}
    pod_id = data.get("id") or data.get("podId")
    if not pod_id:
        raise SystemExit("Could not find pod id in create response")
    save_current_pod_state(str(pod_id))
    return str(pod_id)


def ensure_pod_ready_from_config(config: dict[str, str]) -> str:
    pod_name = config.get("POD_NAME", "pgolf-working")
    wait_for_ssh = config.get("WAIT_FOR_SSH", "1") == "1"
    attempts = int(config.get("WAIT_FOR_SSH_ATTEMPTS", "60"))
    interval = int(config.get("WAIT_FOR_SSH_INTERVAL_SECONDS", "10"))
    reusable = find_reusable_pod(pod_name)
    if reusable:
        reusable_id = str(reusable["id"])
        if str(reusable.get("desiredStatus")) == "EXITED":
            if not start_pod(reusable_id):
                reusable = None
            else:
                reusable["desiredStatus"] = "RUNNING"
        if reusable is not None:
            if wait_for_ssh:
                wait_for_ssh_ready(reusable_id, attempts, interval)
            save_current_pod_state(reusable_id)
            return reusable_id
    pod_id = create_pod(config)
    if wait_for_ssh:
        wait_for_ssh_ready(pod_id, attempts, interval)
    return pod_id


def ensure_pod_ready(pod_config_path: Path) -> str:
    return ensure_pod_ready_from_config(parse_env_file(pod_config_path))


def remote_ssh_base(pod_id: str) -> list[str]:
    host, port, key_path, _ssh_command = ssh_parts(pod_id)
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        key_path,
        "-p",
        port,
        f"root@{host}",
    ]


def ssh_run(pod_id: str, command: str, *, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return run([*remote_ssh_base(pod_id), command], check=check, capture_output=capture_output)


def sync_target(local_path: Path, remote_path: str, *, exclude: list[str] | None = None, delete: bool = True, pod_id: str) -> None:
    host, port, key_path, _ = ssh_parts(pod_id)
    ssh_run(pod_id, f"mkdir -p {shlex.quote(remote_path)}", check=True, capture_output=True)
    cmd = [
        "rsync",
        "-az",
        "--no-owner",
        "--no-group",
    ]
    if delete:
        cmd.append("--delete")
    for pattern in exclude or []:
        cmd.extend(["--exclude", pattern])
    cmd.extend(
        [
            "-e",
            f"ssh -o StrictHostKeyChecking=no -i {shlex.quote(key_path)} -p {shlex.quote(port)}",
            f"{str(local_path).rstrip('/')}/",
            f"root@{host}:{remote_path.rstrip('/')}/",
        ]
    )
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout or "rsync failed")


def sync_repo_snapshot(pod_id: str) -> None:
    if not shutil_which("rsync"):
        raise SystemExit("rsync is required for this workflow")
    eprint(f"==> Syncing repo snapshot to pod {pod_id}")
    sync_target(
        REPO_ROOT_LOCAL / "data",
        f"{REPO_ROOT_REMOTE}/data",
        exclude=["datasets"],
        pod_id=pod_id,
    )
    sync_target(
        RECORD_DIR,
        f"{REPO_ROOT_REMOTE}/{RECORD_ROOT_REMOTE}",
        exclude=["runpod/state", "runpod/secrets.env", "logs", "__pycache__", ".DS_Store"],
        pod_id=pod_id,
    )


def remote_training_running(pod_id: str) -> bool:
    result = ssh_run(
        pod_id,
        "python3 - <<'PY'\n"
        "import pathlib, subprocess\n"
        "lock = pathlib.Path('/workspace/pgolf-runtime/train.pid')\n"
        "if lock.exists():\n"
        "    pid = lock.read_text().strip()\n"
        "    if pid.isdigit() and subprocess.run(['kill', '-0', pid]).returncode == 0:\n"
        "        raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
        "PY",
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def remote_bootstrap(pod_id: str, train_config_path: Path, env_overrides: dict[str, str] | None = None) -> None:
    eprint("==> Running remote bootstrap")
    export_parts = []
    for key, value in (env_overrides or {}).items():
        export_parts.append(f"export {key}={shlex.quote(value)};")
    ssh_cmd = (
        f"cd {shlex.quote(REPO_ROOT_REMOTE)} && "
        f"{' '.join(export_parts)} "
        f"bash {shlex.quote(f'{REPO_ROOT_REMOTE}/{RECORD_ROOT_REMOTE}/runpod/remote_bootstrap.sh')} "
        f"{shlex.quote(f'{REPO_ROOT_REMOTE}/{RECORD_ROOT_REMOTE}/runpod/{train_config_path.name}')}"
    )
    exit_code = stream([*remote_ssh_base(pod_id), ssh_cmd])
    if exit_code != 0:
        raise SystemExit(exit_code)


def load_secrets() -> dict[str, str]:
    secrets_path = SCRIPT_DIR / "secrets.env"
    if not secrets_path.exists():
        return {}
    return parse_env_file(secrets_path)


def local_git_commit() -> str:
    try:
        return run(["git", "-C", str(REPO_ROOT_LOCAL), "rev-parse", "HEAD"]).stdout.strip()
    except subprocess.CalledProcessError:
        return "snapshot-no-git"


def remote_train(pod_id: str, train_config_path: Path, run_id: str, extra_env: dict[str, str], auto_stop: bool = True) -> int:
    secrets = load_secrets()
    exports: dict[str, str] = {}
    if run_id:
        exports["RUN_ID"] = run_id
        exports["WANDB_RUN_NAME"] = run_id
    if api_key := secrets.get("WANDB_API_KEY"):
        exports["WANDB_API_KEY"] = api_key
    exports["RUNPOD_POD_ID"] = pod_id
    exports["SOURCE_GIT_COMMIT"] = local_git_commit()
    exports.update(extra_env)
    export_parts = [f"export {key}={shlex.quote(value)};" for key, value in exports.items()]
    remote_cmd = (
        f"mkdir -p /workspace/pgolf-runtime && "
        f"cd {shlex.quote(REPO_ROOT_REMOTE)} && "
        f"{' '.join(export_parts)} "
        f"bash {shlex.quote(f'{REPO_ROOT_REMOTE}/{RECORD_ROOT_REMOTE}/runpod/remote_train.sh')} "
        f"{shlex.quote(f'{REPO_ROOT_REMOTE}/{RECORD_ROOT_REMOTE}/runpod/{train_config_path.name}')}"
    )
    try:
        return stream([*remote_ssh_base(pod_id), remote_cmd])
    finally:
        if auto_stop:
            eprint(f"==> Stopping pod {pod_id}")
            subprocess.run(["runpodctl", "pod", "stop", pod_id], text=True, capture_output=True)


def shutil_which(binary: str) -> str | None:
    from shutil import which

    return which(binary)


def summarize_runs(runs_dir: str) -> int:
    return stream([str(MINICONDA_PYTHON), str(SCRIPT_DIR / "summarize_runs.py"), runs_dir], cwd=SCRIPT_DIR)


def status_command(pod_id: str | None) -> int:
    pod_id = resolve_pod_id(pod_id)
    pod = pod_get_json(pod_id)
    print("==> Pod")
    for key in ("id", "name", "desiredStatus", "gpuCount", "costPerHr", "imageName"):
        value = pod.get(key)
        if value is not None:
            print(f"{key}: {value}")
    ssh = pod.get("ssh") or {}
    if ssh.get("ip") and ssh.get("port"):
        print(f"ssh: {ssh['ip']}:{ssh['port']}")
    if pod.get("desiredStatus") != "RUNNING":
        return 0
    print("\n==> SSH")
    info = wait_for_ssh_ready(pod_id, 6, 5)
    print(json.dumps(info, indent=2, sort_keys=True))
    print("\n==> Remote")
    cmd = (
        "set -euo pipefail\n"
        f"if [[ -d {shlex.quote(REPO_ROOT_REMOTE)} ]]; then cd {shlex.quote(REPO_ROOT_REMOTE)}; else echo 'workspace_missing'; exit 0; fi\n"
        "printf 'latest_run_dir: '\n"
        "ls -1dt logs/record_runs/* 2>/dev/null | head -n 1 || echo none\n"
        "printf 'training_processes:\\n'\n"
        "ps -ef | grep -E 'train_gpt.py|torchrun|python -m torch.distributed.run' | grep -v grep || true\n"
        "latest_run=\"$(ls -1dt logs/record_runs/* 2>/dev/null | head -n 1 || true)\"\n"
        "if [[ -n \"${latest_run}\" ]]; then\n"
        "  echo 'latest_log_tail:'\n"
        "  tail -n 20 \"${latest_run}/train.log\" || true\n"
        "  latest_wandb_url=\"$(grep -Eo 'https://wandb.ai/[^ ]+' \"${latest_run}/console.log\" 2>/dev/null | tail -n 1 || true)\"\n"
        "  if [[ -n \"${latest_wandb_url}\" ]]; then printf 'wandb_url: %s\\n' \"${latest_wandb_url}\"; fi\n"
        "fi\n"
    )
    print(ssh_run(pod_id, cmd, capture_output=True).stdout, end="")
    return 0


def ssh_command(pod_id: str | None) -> int:
    pod_id = resolve_pod_id(pod_id)
    _host, _port, _key, ssh_command = ssh_parts(pod_id)
    return subprocess.call(ssh_command, shell=True)


def stop_command(pod_id: str | None) -> int:
    pod_id = resolve_pod_id(pod_id)
    result = run(["runpodctl", "pod", "stop", pod_id], capture_output=True)
    print(result.stdout, end="")
    return 0


def delete_command(pod_id: str | None) -> int:
    pod_id = resolve_pod_id(pod_id)
    result = run(["runpodctl", "pod", "delete", pod_id], capture_output=True)
    print(result.stdout, end="")
    clear_current_pod_state(pod_id)
    return 0


def delete_pod_best_effort(pod_id: str | None) -> None:
    if not pod_id:
        return
    subprocess.run(["runpodctl", "pod", "stop", pod_id], text=True, capture_output=True)
    subprocess.run(["runpodctl", "pod", "delete", pod_id], text=True, capture_output=True)
    clear_current_pod_state(pod_id)


def run_command(train_config: Path, pod_config: Path, run_id: str, extra_env_raw: str) -> int:
    extra_env = parse_extra_env(extra_env_raw)
    bootstrap_overrides: dict[str, str] = {}
    pod_values = parse_env_file(pod_config)
    try:
        pod_id = ensure_pod_ready_from_config(pod_values)
    except TimeoutError:
        if pod_values.get("RUNPOD_IMAGE", "").strip() and pod_values.get("RUNPOD_TEMPLATE_ID", "").strip():
            failed_pod_id = CURRENT_POD_ID_PATH.read_text(encoding="utf-8").strip() if CURRENT_POD_ID_PATH.exists() else None
            eprint("==> Custom image pod never became SSH-ready; deleting it and retrying with the stock template")
            delete_pod_best_effort(failed_pod_id)
            fallback_values = dict(pod_values)
            fallback_values["RUNPOD_IMAGE"] = ""
            bootstrap_overrides["SKIP_PIP_INSTALL"] = "0"
            pod_id = ensure_pod_ready_from_config(fallback_values)
        else:
            raise
    if remote_training_running(pod_id):
        raise SystemExit(f"A training process is already running on pod {pod_id}. Use 'just status' to inspect it.")
    sync_repo_snapshot(pod_id)
    remote_bootstrap(pod_id, train_config, bootstrap_overrides)
    return remote_train(pod_id, train_config, run_id, {**bootstrap_overrides, **extra_env})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter Golf Runpod orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the default 2x5090 experiment")
    run_parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    run_parser.add_argument("--pod-config", type=Path, default=DEFAULT_POD_CONFIG)
    run_parser.add_argument("--run-id", default="")
    run_parser.add_argument("--extra-env", default="")

    smoke_parser = subparsers.add_parser("smoke", help="Run the 2x5090 smoke config")
    smoke_parser.add_argument("--train-config", type=Path, default=DEFAULT_SMOKE_TRAIN_CONFIG)
    smoke_parser.add_argument("--pod-config", type=Path, default=DEFAULT_POD_CONFIG)
    smoke_parser.add_argument("--run-id", default="")
    smoke_parser.add_argument("--extra-env", default="")

    record_parser = subparsers.add_parser("record", help="Run the 8xH100 record config")
    record_parser.add_argument("--train-config", type=Path, default=DEFAULT_RECORD_TRAIN_CONFIG)
    record_parser.add_argument("--pod-config", type=Path, default=DEFAULT_RECORD_POD_CONFIG)
    record_parser.add_argument("--run-id", default="")
    record_parser.add_argument("--extra-env", default="")

    status_parser = subparsers.add_parser("status", help="Show pod/run status")
    status_parser.add_argument("--pod-id", default=None)

    ssh_parser = subparsers.add_parser("ssh", help="Open SSH to current pod")
    ssh_parser.add_argument("--pod-id", default=None)

    stop_parser = subparsers.add_parser("stop", help="Stop current pod")
    stop_parser.add_argument("--pod-id", default=None)

    delete_parser = subparsers.add_parser("delete", help="Delete current pod")
    delete_parser.add_argument("--pod-id", default=None)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize pulled runs")
    summarize_parser.add_argument("--runs-dir", default="/workspace/parameter-golf/logs/record_runs")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in {"run", "smoke", "record"}:
        return run_command(args.train_config, args.pod_config, args.run_id, args.extra_env)
    if args.command == "status":
        return status_command(args.pod_id)
    if args.command == "ssh":
        return ssh_command(args.pod_id)
    if args.command == "stop":
        return stop_command(args.pod_id)
    if args.command == "delete":
        return delete_command(args.pod_id)
    if args.command == "summarize":
        return summarize_runs(args.runs_dir)
    raise SystemExit(f"Unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
