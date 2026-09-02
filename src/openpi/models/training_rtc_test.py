import jax
import jax.numpy as jnp

from openpi.models.training_rtc import apply_clean_prefix, postfix_mse, prepare_training_rtc, sample_prefix_lengths


def test_clean_prefix_replaces_only_committed_h20_steps():
    noisy = jnp.zeros((3, 20, 7), dtype=jnp.float32)
    clean = jnp.ones_like(noisy)

    conditioned = apply_clean_prefix(noisy, clean, jnp.array([0, 1, 4], dtype=jnp.int32))

    assert conditioned.shape == (3, 20, 7)
    assert jnp.all(conditioned[0] == 0)
    assert jnp.all(conditioned[1, :1] == 1)
    assert jnp.all(conditioned[1, 1:] == 0)
    assert jnp.all(conditioned[2, :4] == 1)
    assert jnp.all(conditioned[2, 4:] == 0)


def test_postfix_loss_masks_clean_prefix_and_preserves_disabled_parity():
    predicted = jnp.arange(3 * 20 * 7, dtype=jnp.float32).reshape(3, 20, 7)
    target = jnp.zeros_like(predicted)

    disabled = postfix_mse(predicted, target, None)
    baseline = jnp.mean(jnp.square(predicted - target), axis=-1)
    assert jnp.array_equal(disabled, baseline)

    masked = postfix_mse(predicted, target, jnp.array([0, 1, 4], dtype=jnp.int32))
    assert jnp.all(masked[0] == baseline[0])
    assert jnp.all(masked[1, :1] == 0)
    assert jnp.all(masked[1, 1:] == baseline[1, 1:])
    assert jnp.all(masked[2, :4] == 0)
    assert jnp.all(masked[2, 4:] == baseline[2, 4:])


def test_postfix_loss_is_finite_for_h20_7d_and_jit_gradients():
    predicted = jnp.ones((3, 20, 7), dtype=jnp.float32)
    target = jnp.zeros_like(predicted)
    prefix_lengths = jnp.array([0, 1, 4], dtype=jnp.int32)

    def objective(x):
        return jnp.mean(postfix_mse(x, target, prefix_lengths))

    value, gradient = jax.jit(jax.value_and_grad(objective))(predicted)

    assert jnp.isfinite(value)
    assert gradient.shape == predicted.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient[1, :1] == 0)
    assert jnp.all(gradient[2, :4] == 0)


def test_prefix_sampler_is_fixed_seed_deterministic_and_bounded():
    first = sample_prefix_lengths(jax.random.key(17), batch_size=64, max_delay=4)
    second = sample_prefix_lengths(jax.random.key(17), batch_size=64, max_delay=4)

    assert first.shape == (64,)
    assert jnp.array_equal(first, second)
    assert jnp.all((first >= 0) & (first <= 4))


def test_training_rtc_flow_matching_keeps_clean_prefix_and_masks_only_prefix_loss():
    from openpi.models.training_rtc import prepare_training_rtc

    actions = jnp.arange(2 * 20 * 7, dtype=jnp.float32).reshape(2, 20, 7)
    noise = actions + 1000.0
    time = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    prefix_lengths = jnp.asarray([1, 4], dtype=jnp.int32)

    conditioned, target_velocity, loss_mask = prepare_training_rtc(
        actions,
        noise,
        time,
        prefix_lengths,
    )

    expected_noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    assert conditioned.shape == (2, 20, 7)
    assert target_velocity.shape == (2, 20, 7)
    assert loss_mask.shape == (2, 20)
    assert jnp.all(conditioned[0, :1] == actions[0, :1])
    assert jnp.all(conditioned[1, :4] == actions[1, :4])
    assert jnp.all(conditioned[0, 1:] == expected_noisy[0, 1:])
    assert jnp.all(conditioned[1, 4:] == expected_noisy[1, 4:])
    assert jnp.all(target_velocity == noise - actions)
    assert jnp.all(loss_mask == jnp.asarray([[False] + [True] * 19, [False] * 4 + [True] * 16]))


def test_training_rtc_uses_only_physical_dims_when_actions_are_padded_to_latent_32():
    actions = jnp.zeros((1, 20, 32), dtype=jnp.float32)
    noise = jnp.ones_like(actions)
    time = jnp.asarray([0.5], dtype=jnp.float32)
    prefix_lengths = jnp.asarray([4], dtype=jnp.int32)

    conditioned, target_velocity, loss_mask = prepare_training_rtc(
        actions,
        noise,
        time,
        prefix_lengths,
        physical_action_dim=7,
    )

    # RTC conditions only the physical Piper dimensions; latent padding remains
    # ordinary noisy interpolation and contributes no RTC loss.
    assert jnp.all(conditioned[0, :4, :7] == 0)
    assert jnp.all(conditioned[0, :4, 7:] == 0.5)
    assert jnp.all(loss_mask[0, :4] == jnp.asarray(False))
    assert jnp.all(target_velocity == 1)

    predicted = jnp.zeros_like(actions)
    physical_only = postfix_mse(
        predicted,
        target_velocity,
        prefix_lengths,
        physical_action_dim=7,
    )
    assert jnp.all(physical_only[0, :4] == 0)
    assert jnp.all(physical_only[0, 4:] == 1)


def test_quality58_config_keeps_physical_7d_but_uses_pretrained_latent_32d():
    from openpi.training import config as training_config
    from openpi.transforms import PadStatesAndActions

    config = training_config.get_config("pi05_piper_joint_rtc_h20_quality58_finetune")
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 20
    assert config.model.training_rtc_action_dim == 7

    data_config = config.data.create(config.assets_dirs, config.model)
    pad = [t for t in data_config.model_transforms.inputs if isinstance(t, PadStatesAndActions)]
    assert len(pad) == 1
    assert pad[0].model_action_dim == 32
