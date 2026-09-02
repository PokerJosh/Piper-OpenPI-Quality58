"""Training-time RTC helpers for fixed-shape action chunks."""

import jax
import jax.numpy as jnp


def _prefix_lengths(prefix_lengths, *, batch_size: int, horizon: int) -> jax.Array:
    lengths = jnp.asarray(prefix_lengths, dtype=jnp.int32)
    if lengths.ndim == 0:
        lengths = jnp.broadcast_to(lengths, (batch_size,))
    elif lengths.shape != (batch_size,):
        raise ValueError(f"prefix_lengths must have shape ({batch_size},), got {lengths.shape}")
    return jnp.clip(lengths, 0, horizon)


def prefix_mask(prefix_lengths, *, batch_size: int, horizon: int) -> jax.Array:
    """Return a bool [B, H] mask identifying committed prefix actions."""
    lengths = _prefix_lengths(prefix_lengths, batch_size=batch_size, horizon=horizon)
    return jnp.arange(horizon, dtype=jnp.int32)[None, :] < lengths[:, None]


def apply_clean_prefix(noisy_actions, clean_actions, prefix_lengths):
    """Replace each sample's committed prefix with its clean action sequence."""
    if noisy_actions.shape != clean_actions.shape:
        raise ValueError(f"noisy_actions and clean_actions must have the same shape, got {noisy_actions.shape}")
    if noisy_actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, H, A], got {noisy_actions.shape}")
    committed = prefix_mask(
        prefix_lengths,
        batch_size=noisy_actions.shape[0],
        horizon=noisy_actions.shape[1],
    )[..., None]
    return jnp.where(committed, clean_actions, noisy_actions)


def prepare_training_rtc(actions, noise, time, prefix_lengths, physical_action_dim=None):
    """Build the fixed-shape training-time RTC flow-matching objective inputs.

    The committed prefix is clean action data, while the postfix remains the
    ordinary noisy interpolation. The returned loss mask is false only for the
    committed prefix, so a clean prefix cannot create an artificial loss or
    gradient at an inter-chunk boundary.
    """
    if actions.shape != noise.shape:
        raise ValueError(f"actions and noise must have the same shape, got {actions.shape} and {noise.shape}")
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, H, A], got {actions.shape}")

    batch_size, horizon, _ = actions.shape
    time = jnp.asarray(time, dtype=actions.dtype)
    if time.ndim == 0:
        time = jnp.broadcast_to(time, (batch_size,))
    elif time.shape != (batch_size,):
        raise ValueError(f"time must have shape ({batch_size},), got {time.shape}")

    time_expanded = time[:, None, None]
    noisy_interpolation = time_expanded * noise + (1.0 - time_expanded) * actions
    if physical_action_dim is None:
        physical_action_dim = actions.shape[-1]
    if not (0 < physical_action_dim <= actions.shape[-1]):
        raise ValueError(
            f"physical_action_dim must be between 1 and action_dim, got {physical_action_dim}"
        )
    committed = prefix_mask(prefix_lengths, batch_size=batch_size, horizon=horizon)
    physical = jnp.arange(actions.shape[-1], dtype=jnp.int32) < physical_action_dim
    clean_mask = committed[..., None] & physical[None, None, :]
    conditioned_actions = jnp.where(clean_mask, actions, noisy_interpolation)
    return conditioned_actions, noise - actions, jnp.logical_not(committed)


def postfix_mse(predicted_velocity, target_velocity, prefix_lengths, physical_action_dim=None):
    """Compute per-step MSE, masking committed prefix steps from the loss."""
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError(
            "predicted_velocity and target_velocity must have the same shape, "
            f"got {predicted_velocity.shape} and {target_velocity.shape}"
        )
    if predicted_velocity.ndim != 3:
        raise ValueError(f"velocity tensors must have shape [B, H, A], got {predicted_velocity.shape}")

    if physical_action_dim is None:
        physical_action_dim = predicted_velocity.shape[-1]
    if not (0 < physical_action_dim <= predicted_velocity.shape[-1]):
        raise ValueError(
            f"physical_action_dim must be between 1 and action_dim, got {physical_action_dim}"
        )
    predicted_velocity = predicted_velocity[..., :physical_action_dim]
    target_velocity = target_velocity[..., :physical_action_dim]
    mse = jnp.mean(jnp.square(predicted_velocity - target_velocity), axis=-1)
    if prefix_lengths is None:
        return mse
    committed = prefix_mask(
        prefix_lengths,
        batch_size=predicted_velocity.shape[0],
        horizon=predicted_velocity.shape[1],
    )
    return jnp.where(committed, 0.0, mse)


def sample_prefix_lengths(rng, *, batch_size: int, max_delay: int) -> jax.Array:
    """Sample a fixed-shape inclusive prefix length in [0, max_delay]."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")
    return jax.random.randint(rng, (batch_size,), minval=0, maxval=max_delay + 1, dtype=jnp.int32)
