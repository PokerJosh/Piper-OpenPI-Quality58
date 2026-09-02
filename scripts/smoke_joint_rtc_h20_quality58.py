"""Run the exact five-step Quality58 H20 training smoke on an RTX5090.

This script intentionally uses the production train-state and train-step code,
but never initializes checkpoints/W&B and never saves anything. It is a gate,
not a formal training launcher.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp

import openpi.training.config as config_lib
import openpi.training.data_loader as data_loader
import openpi.training.sharding as sharding

# scripts/ is not a package; import the production train helpers directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import init_train_state, train_step  # noqa: E402

CONFIG_NAME = "pi05_piper_joint_rtc_h20_quality58_finetune"
SMOKE_STEPS = 5


def _run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
        return (result.stdout + result.stderr).strip()
    except Exception as exc:  # pragma: no cover - diagnostics must not mask the gate result.
        return f"command failed: {type(exc).__name__}: {exc}"


def _gpu_snapshot() -> dict[str, str]:
    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "nvidia_smi_query": query,
        "nvidia_smi_full": _run_command(["nvidia-smi"]),
    }


def _host_rss_mib() -> float:
    # Linux reports KiB in ru_maxrss.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _is_cuda_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}".lower()
    return any(token in text for token in ("cuda out of memory", "cuda oom", "resource exhausted", "out of memory"))


def _assert_rtx5090() -> dict[str, object]:
    devices = jax.devices()
    if jax.default_backend() not in {"gpu", "cuda"} or not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"RTX5090 gate requires GPU JAX backend, got backend={jax.default_backend()} devices={devices}")
    gpu_info = _gpu_snapshot()
    if "5090" not in gpu_info["nvidia_smi_query"]:
        raise RuntimeError(f"RTX5090 gate could not confirm an RTX5090 from nvidia-smi: {gpu_info['nvidia_smi_query']}")
    return {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
        **gpu_info,
    }


def _run_once(batch_size: int, fallback_used: bool) -> int:
    started = time.perf_counter()
    report: dict[str, object] = {
        "config_name": CONFIG_NAME,
        "requested_batch_size": batch_size,
        "fallback_used": fallback_used,
        "steps_requested": SMOKE_STEPS,
        "checkpoint_save": False,
        "orbax_save": False,
        "wandb_enabled": False,
        "robot_connected": False,
        "can_write": False,
        "losses": [],
        "grad_norms": [],
        "step_seconds": [],
        "host_rss_peak_mib": _host_rss_mib(),
        "gpu_before": {},
    }

    try:
        gpu_info = _assert_rtx5090()
        report["gpu_before"] = gpu_info
        config = config_lib.get_config(CONFIG_NAME)
        if config.batch_size != 16 or config.model.action_horizon != 20 or config.model.action_dim != 32:
            raise RuntimeError(f"unexpected exact config values: batch_size={config.batch_size}, model={config.model}")
        if (
            not config.model.training_rtc
            or config.model.training_rtc_max_delay != 4
            or config.model.training_rtc_action_dim != 7
        ):
            raise RuntimeError("Quality58 config is not using H20 training-time RTC max delay 4")
        if batch_size not in (16, 8):
            raise ValueError(f"smoke batch size must be 16 or the single OOM fallback 8, got {batch_size}")

        mesh = sharding.make_mesh(config.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        smoke_config = config if batch_size == config.batch_size else config_lib.dataclasses.replace(config, batch_size=batch_size)

        data_loader_obj = data_loader.create_data_loader(
            smoke_config,
            sharding=data_sharding,
            shuffle=True,
            num_batches=SMOKE_STEPS,
        )
        data_iter = iter(data_loader_obj)
        first_batch = next(data_iter)
        report["dataset_class"] = type(data_loader_obj).__name__
        report["dataset_length"] = len(data_loader_obj._data_loader._data_loader.dataset)  # noqa: SLF001
        report["batch_shapes"] = {
            "state": list(first_batch[0].state.shape),
            "actions": list(first_batch[1].shape),
        }

        rng = jax.random.key(config.seed)
        train_rng, init_rng = jax.random.split(rng)
        init_started = time.perf_counter()
        train_state, train_state_sharding = init_train_state(smoke_config, init_rng, mesh, resume=False)
        jax.block_until_ready(train_state)
        report["init_seconds"] = time.perf_counter() - init_started

        ptrain_step = jax.jit(
            functools.partial(train_step, smoke_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

        for step in range(SMOKE_STEPS):
            batch = first_batch if step == 0 else next(data_iter)
            step_started = time.perf_counter()
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            jax.block_until_ready((train_state, info))
            elapsed = time.perf_counter() - step_started
            loss = float(jax.device_get(info["loss"]))
            grad_norm = float(jax.device_get(info["grad_norm"]))
            if not (jnp.isfinite(loss) and jnp.isfinite(grad_norm)):
                raise RuntimeError(f"non-finite smoke metrics at step {step}: loss={loss}, grad_norm={grad_norm}")
            report["losses"].append(loss)
            report["grad_norms"].append(grad_norm)
            report["step_seconds"].append(elapsed)
            report["host_rss_peak_mib"] = max(float(report["host_rss_peak_mib"]), _host_rss_mib())
            report[f"gpu_step_{step}"] = _gpu_snapshot()

        report["status"] = "PASS"
        report["steps_completed"] = SMOKE_STEPS
        report["elapsed_seconds"] = time.perf_counter() - started
        report["gpu_after"] = _gpu_snapshot()
        print(json.dumps(report, indent=2))
        return 0
    except BaseException as exc:
        report["status"] = "FAIL"
        report["steps_completed"] = len(report["losses"])
        report["elapsed_seconds"] = time.perf_counter() - started
        report["exception"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["gpu_after"] = _gpu_snapshot()
        report["host_rss_peak_mib"] = max(float(report["host_rss_peak_mib"]), _host_rss_mib())
        print(json.dumps(report, indent=2), file=sys.stderr)
        if batch_size == 16 and not fallback_used and _is_cuda_oom(exc):
            print("BATCH16_CUDA_OOM=YES; starting the single permitted clean-process batch8 fallback", file=sys.stderr)
            env = dict(os.environ)
            env["OPENPI_QUALITY58_SMOKE_FALLBACK"] = "1"
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--batch-size", "8", "--fallback-used"],
                check=False,
                env=env,
            ).returncode
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fallback-used", action="store_true")
    args = parser.parse_args()
    return _run_once(args.batch_size, args.fallback_used)


if __name__ == "__main__":
    raise SystemExit(main())
