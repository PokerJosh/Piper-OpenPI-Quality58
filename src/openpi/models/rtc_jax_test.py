"""Focused contract tests for the minimal JAX RTC guidance adapter.

These tests intentionally exercise the small, model-independent RTC boundary so
that the expensive Pi0 checkpoint is not needed to validate shape, masking, and
autodiff semantics.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models.rtc_jax import (
    make_rtc_prefix_weights,
    prepare_rtc_prefix,
    rtc_guided_velocity,
)


def _base_velocity(x):
    """Small nonlinear flow field with a nontrivial VJP."""

    return 0.1 * x + 0.02 * jnp.square(x)


def _inputs(batch=2, horizon=10, action_dim=7):
    x_t = jnp.linspace(-0.8, 0.8, batch * horizon * action_dim, dtype=jnp.float32).reshape(
        batch, horizon, action_dim
    )
    prev = jnp.flip(x_t, axis=1) + 0.35
    return x_t, prev


def test_sample_actions_keeps_optional_rtc_contract():
    signature = inspect.signature(pi0.Pi0.sample_actions)
    for name in (
        "rtc_prev_chunk_left_over",
        "rtc_inference_delay",
        "rtc_execution_horizon",
        "rtc_prefix_weights",
        "rtc_max_guidance_weight",
    ):
        assert name in signature.parameters


def test_none_prefix_is_exact_baseline_and_does_not_call_vjp(monkeypatch):
    x_t, _ = _inputs()
    expected = _base_velocity(x_t)

    def fail_vjp(*_args, **_kwargs):
        raise AssertionError("disabled RTC must not construct a VJP")

    monkeypatch.setattr(jax, "vjp", fail_vjp)
    actual = rtc_guided_velocity(
        _base_velocity,
        x_t,
        time=jnp.asarray(0.5, dtype=jnp.float32),
        prev_chunk_left_over=None,
        inference_delay=3,
        execution_horizon=10,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_prev_none_is_baseline_even_when_other_rtc_args_are_present():
    x_t, _ = _inputs()
    expected = _base_velocity(x_t)
    actual = rtc_guided_velocity(
        _base_velocity,
        x_t,
        time=0.5,
        prev_chunk_left_over=None,
        inference_delay=3,
        execution_horizon=10,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_short_prefix_is_padded_to_fixed_shape_with_mask():
    short = jnp.ones((2, 4, 7), dtype=jnp.float32)
    padded, mask = prepare_rtc_prefix(short, batch_size=2, horizon=10, action_dim=7)
    assert padded.shape == (2, 10, 7)
    assert mask.shape == (2, 10, 1)
    np.testing.assert_array_equal(np.asarray(padded[:, :4]), np.ones((2, 4, 7), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(padded[:, 4:]), np.zeros((2, 6, 7), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(mask[0, :, 0]), np.array([1] * 4 + [0] * 6, dtype=bool))


def test_prefix_guidance_changes_velocity_and_preserves_action_shape():
    x_t, prev = _inputs()
    actual = rtc_guided_velocity(
        _base_velocity,
        x_t,
        time=0.5,
        prev_chunk_left_over=prev,
        inference_delay=3,
        execution_horizon=10,
    )
    baseline = _base_velocity(x_t)
    assert actual.shape == (2, 10, 7)
    assert not np.array_equal(np.asarray(actual), np.asarray(baseline))


def test_same_noise_and_prefix_are_deterministic():
    x_t, prev = _inputs()
    kwargs = dict(
        time=0.25,
        prev_chunk_left_over=prev,
        inference_delay=3,
        execution_horizon=10,
    )
    first = rtc_guided_velocity(_base_velocity, x_t, **kwargs)
    second = rtc_guided_velocity(_base_velocity, x_t, **kwargs)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_guidance_is_finite_for_delay_three_and_horizon_ten():
    x_t, prev = _inputs()
    actual = rtc_guided_velocity(
        _base_velocity,
        x_t,
        time=1.0,
        prev_chunk_left_over=prev,
        inference_delay=3,
        execution_horizon=10,
    )
    assert np.isfinite(np.asarray(actual)).all()


def test_execution_horizon_and_delay_are_validated():
    with pytest.raises(ValueError):
        make_rtc_prefix_weights(horizon=10, inference_delay=11, execution_horizon=10)
    with pytest.raises(ValueError):
        make_rtc_prefix_weights(horizon=10, inference_delay=3, execution_horizon=0)


def test_prefix_weights_have_fixed_shape_and_delay_three():
    weights = make_rtc_prefix_weights(horizon=10, inference_delay=3, execution_horizon=10)
    assert weights.shape == (10, 1)
    # The committed delay prefix is fully weighted and the final point is zero.
    np.testing.assert_allclose(np.asarray(weights[:3, 0]), np.ones(3), rtol=0, atol=0)
    assert float(weights[3, 0]) < 1.0
    # With horizon==execution_horizon the final point is the last interior
    # linear weight (the post-horizon zero tail is empty).
    assert float(weights[-1, 0]) == pytest.approx(0.125)


def test_jit_compiles_with_fixed_prefix_shape():
    x_t, prev = _inputs(batch=1)
    mask = jnp.ones((1, 10, 1), dtype=bool)

    @jax.jit
    def run(x, p, m):
        return rtc_guided_velocity(
            _base_velocity,
            x,
            time=jnp.asarray(0.5, dtype=jnp.float32),
            prev_chunk_left_over=p,
            prefix_mask=m,
            inference_delay=3,
            execution_horizon=10,
        )

    out = run(x_t, prev, mask)
    assert out.shape == (1, 10, 7)
    assert np.isfinite(np.asarray(out)).all()


def test_jit_accepts_dynamic_delay_and_horizon_scalars():
    x_t, prev = _inputs(batch=1)
    mask = jnp.ones((1, 10, 1), dtype=bool)

    @jax.jit
    def run(x, p, m, delay, horizon):
        return rtc_guided_velocity(
            _base_velocity,
            x,
            time=jnp.asarray(0.5, dtype=jnp.float32),
            prev_chunk_left_over=p,
            prefix_mask=m,
            inference_delay=delay,
            execution_horizon=horizon,
        )

    out = run(x_t, prev, mask, jnp.asarray(3, dtype=jnp.int32), jnp.asarray(10, dtype=jnp.int32))
    assert out.shape == (1, 10, 7)
    assert np.isfinite(np.asarray(out)).all()


def test_gripper_dimension_uses_same_model_space_guidance():
    x_t, prev = _inputs(batch=1)
    baseline = _base_velocity(x_t)
    # Joint model space is [six delta joints, one absolute gripper] and RTC
    # must not silently drop the seventh model-space dimension.
    prev = prev.at[:, :, 6].add(1.0)
    actual = rtc_guided_velocity(
        _base_velocity,
        x_t,
        time=0.5,
        prev_chunk_left_over=prev,
        inference_delay=3,
        execution_horizon=10,
    )
    assert actual.shape[-1] == 7
    assert not np.array_equal(np.asarray(actual[..., 6]), np.asarray(baseline[..., 6]))
