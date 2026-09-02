"""Minimal JAX implementation of Real-Time Chunking (RTC) guidance.

The module deliberately contains only model-space prefix preparation and the
flow-matching guidance step.  Scheduling, queues, and robot execution remain
outside the model.
"""

from __future__ import annotations

from collections.abc import Callable
import numbers

import jax
import jax.numpy as jnp


def prepare_rtc_prefix(
    prev_chunk_left_over,
    *,
    batch_size: int,
    horizon: int,
    action_dim: int,
):
    """Pad/broadcast a previous model-space prefix to a fixed ``[B,H,A]``.

    The mask is kept separate so right-padding never becomes a false RTC
    target.  This wrapper is intentionally called outside the denoising loop;
    the loop itself only sees fixed-shape arrays.
    """

    if horizon <= 0 or action_dim <= 0 or batch_size <= 0:
        raise ValueError("batch_size, horizon, and action_dim must be positive")
    if prev_chunk_left_over is None:
        dtype = jnp.float32
        return (
            jnp.zeros((batch_size, horizon, action_dim), dtype=dtype),
            jnp.zeros((batch_size, horizon, 1), dtype=bool),
        )

    prefix = jnp.asarray(prev_chunk_left_over)
    if prefix.ndim == 2:
        prefix = prefix[None, ...]
    if prefix.ndim != 3:
        raise ValueError("prev_chunk_left_over must have shape [T,A] or [B,T,A]")
    if prefix.shape[-1] != action_dim:
        raise ValueError(f"prefix action dimension {prefix.shape[-1]} != {action_dim}")
    if prefix.shape[0] == 1 and batch_size != 1:
        prefix = jnp.broadcast_to(prefix, (batch_size, prefix.shape[1], action_dim))
    elif prefix.shape[0] != batch_size:
        raise ValueError(f"prefix batch dimension {prefix.shape[0]} != {batch_size}")

    prefix_len = min(prefix.shape[1], horizon)
    if prefix.shape[1] < horizon:
        prefix = jnp.pad(prefix, ((0, 0), (0, horizon - prefix.shape[1]), (0, 0)))
    else:
        prefix = prefix[:, :horizon, :]
    mask = (jnp.arange(horizon)[None, :, None] < prefix_len).astype(bool)
    mask = jnp.broadcast_to(mask, (batch_size, horizon, 1))
    return prefix, mask


def make_rtc_prefix_weights(
    *,
    horizon: int,
    inference_delay: int,
    execution_horizon: int,
):
    """Construct the official linear RTC prefix weights in fixed shape.

    The first ``inference_delay`` positions are fully committed.  The
    remaining overlap is linearly tapered, with positions at and after the
    execution horizon receiving zero weight.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    # Python values are checked eagerly (useful for a scheduler/configuration
    # error).  Traced JAX scalars are validated by the caller's legal scheduler
    # contract and must remain dynamic so one compiled sampler can be reused.
    if isinstance(inference_delay, numbers.Integral) and (inference_delay < 0 or inference_delay > horizon):
        raise ValueError("inference_delay must be in [0, horizon]")
    if isinstance(execution_horizon, numbers.Integral) and (execution_horizon <= 0 or execution_horizon > horizon):
        raise ValueError("execution_horizon must be in [1, horizon]")

    index = jnp.arange(horizon, dtype=jnp.float32)
    start = jnp.asarray(inference_delay, dtype=jnp.float32)
    end = jnp.asarray(execution_horizon, dtype=jnp.float32)
    overlap = end - start
    # This is the LINEAR schedule used by LeRobot RTCProcessor: delay prefix
    # at one, overlap tapered from near one to near zero, then zero.
    tapered = 1.0 - (index - start + 1.0) / (overlap + 1.0)
    weights = jnp.where(
        index < start,
        1.0,
        jnp.where(index < end, jnp.where(overlap > 0, tapered, 1.0), 0.0),
    )
    return weights[:, None]


def _rtc_guidance_weight(time, max_guidance_weight: float):
    """Match the flow-time convention and clamp from the RTC reference."""

    tau = 1.0 - jnp.asarray(time)
    squared_one_minus_tau = (1.0 - tau) ** 2
    inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
    c = (1.0 - tau) / tau
    max_weight = jnp.asarray(max_guidance_weight, dtype=inv_r2.dtype)

    def _finite_or_zero(value):
        return jnp.where(jnp.isnan(value), 0.0, jnp.where(jnp.isposinf(value), max_weight, value))

    inv_r2 = _finite_or_zero(inv_r2)
    c = _finite_or_zero(c)
    guidance = _finite_or_zero(c * inv_r2)
    return jnp.minimum(guidance, max_weight)


def rtc_guided_velocity(
    base_denoise: Callable,
    x_t,
    time,
    prev_chunk_left_over,
    inference_delay: int,
    execution_horizon: int,
    *,
    prefix_mask=None,
    prefix_weights=None,
    max_guidance_weight: float = 10.0,
    enabled: bool = True,
):
    """Apply RTC prefix guidance to one flow-matching denoising step.

    ``x_t`` and ``prev_chunk_left_over`` are both in the model's transformed
    space (for Joint: six delta joints plus absolute gripper, normalized by
    the policy transform).  When RTC is disabled or no prefix is available,
    this calls ``base_denoise`` directly and does not construct a VJP.
    """

    if not enabled or prev_chunk_left_over is None:
        return base_denoise(x_t)

    if x_t.ndim != 3:
        raise ValueError("RTC currently expects fixed [B,H,A] model tensors")
    batch_size, horizon, action_dim = x_t.shape
    if isinstance(inference_delay, numbers.Integral) and (inference_delay < 0 or inference_delay > horizon):
        raise ValueError("inference_delay must be in [0, action horizon]")
    if isinstance(execution_horizon, numbers.Integral) and (execution_horizon <= 0 or execution_horizon > horizon):
        raise ValueError("execution_horizon must be in [1, action horizon]")

    prev = jnp.asarray(prev_chunk_left_over, dtype=x_t.dtype)
    if prev.shape != x_t.shape:
        raise ValueError("prev_chunk_left_over must be padded to x_t's fixed shape")
    if prefix_mask is None:
        prefix_mask = jnp.ones((batch_size, horizon, 1), dtype=x_t.dtype)
    else:
        prefix_mask = jnp.asarray(prefix_mask, dtype=x_t.dtype)
        if prefix_mask.shape != (batch_size, horizon, 1):
            raise ValueError("prefix_mask must have shape [B,H,1]")

    if prefix_weights is None:
        weights = make_rtc_prefix_weights(
            horizon=horizon,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
    else:
        weights = jnp.asarray(prefix_weights, dtype=x_t.dtype)
        if weights.ndim == 1:
            weights = weights[:, None]
        if weights.shape != (horizon, 1):
            raise ValueError("prefix_weights must have shape [H] or [H,1]")
    weights = weights[None, ...] * prefix_mask

    # has_aux keeps the base velocity from being evaluated a second time while
    # still exposing the pullback of x1_t with respect to x_t.
    def x1_and_velocity(x):
        velocity = base_denoise(x)
        return x - jnp.asarray(time) * velocity, velocity

    x1_t, pullback, base_velocity = jax.vjp(x1_and_velocity, x_t, has_aux=True)
    error = (prev - x1_t) * weights
    correction = pullback(error)[0]
    guidance_weight = _rtc_guidance_weight(time, max_guidance_weight)
    return base_velocity - guidance_weight * correction
