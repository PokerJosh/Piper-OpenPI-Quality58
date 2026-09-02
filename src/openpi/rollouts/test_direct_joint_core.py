import numpy as np

from openpi.rollouts.direct_joint_core import (
    LatestOnlyTargetMailbox,
    PhaseAwareExecutor,
    Target,
    gripper_send_decision,
    validate_executor_command,
    validate_raw_target,
)


def test_raw_gate_uses_previous_accepted_target_not_live_feedback():
    feedback = np.zeros(7)
    first = np.array([4.0, 0, 0, 0, 0, 0, 0.02])
    second = np.array([8.5, 0, 0, 0, 0, 0, 0.02])
    assert validate_raw_target(first, feedback) == (True, None)
    assert validate_raw_target(second, first) == (True, None)
    assert validate_raw_target(second, feedback)[0] is False


def test_raw_gate_rejects_bad_shape_nonfinite_and_limits():
    reference = np.zeros(7)
    for target in (np.zeros(6), np.full(7, np.nan), np.array([151, 0, 0, 0, 0, 0, 0.02]), np.array([0, 0, 0, 0, 0, 0, 0.071])):
        assert validate_raw_target(target, reference)[0] is False


def test_raw_gate_does_not_move_farther_outside_deployment_j2_j3_limits():
    assert validate_raw_target(np.array([0, 0, 0, 0, 0, 0, 0.02]), np.array([0, 0.5, 0, 0, 0, 0, 0.02]))[0] is False
    assert validate_raw_target(np.array([0, 0, 0.1, 0, 0, 0, 0.02]), np.zeros(7))[0] is False
    assert validate_raw_target(np.array([0, 1.5, -0.5, 0, 0, 0, 0.02]), np.zeros(7)) == (True, None)


def test_executor_never_exceeds_one_degree_per_tick():
    executor = PhaseAwareExecutor(50.0, 0.50)
    previous = np.zeros(6)
    for _ in range(100):
        current = executor.step(previous, np.full(6, 30.0))
        assert np.max(np.abs(current - previous)) <= 1.0 + 1e-9
        previous = current


def test_executor_brakes_without_overshoot():
    executor = PhaseAwareExecutor(50.0, 0.50)
    current = np.zeros(6)
    target = np.full(6, 2.0)
    for _ in range(200):
        next_command = executor.step(current, target)
        assert np.all(next_command <= target + 1e-9)
        current = next_command
    np.testing.assert_allclose(current, target, atol=1e-9)


def test_executor_command_checks_step_and_velocity_cap():
    assert validate_executor_command(np.zeros(6), np.zeros(6), 0.02, 50.0) == (True, None)
    assert validate_executor_command(np.full(6, 1.01), np.zeros(6), 0.02, 50.0)[0] is False
    assert validate_executor_command(np.full(6, 1.0), np.zeros(6), 0.02, 40.0)[0] is False
    assert validate_executor_command(np.full(6, np.nan), np.zeros(6), 0.02, 50.0)[0] is False


def test_mailbox_overwrites_without_backlog():
    mailbox = LatestOnlyTargetMailbox()
    mailbox.put(Target(np.zeros(7), 1.0, 1))
    mailbox.put(Target(np.ones(7), 2.0, 2))
    assert mailbox.take().generation_id == 2
    assert mailbox.take() is None
    assert mailbox.dropped_count == 1


def test_mailbox_target_values_are_frozen_after_insertion():
    mailbox = LatestOnlyTargetMailbox()
    values = np.zeros(7)
    mailbox.put(Target(values, 1.0, 1))
    values[0] = 9.0

    pending = mailbox.take()
    np.testing.assert_array_equal(pending.values, np.zeros(7))
    with np.testing.assert_raises(ValueError):
        pending.values[0] = 9.0


def test_executor_rejects_unsupported_phase():
    executor = PhaseAwareExecutor()
    with np.testing.assert_raises(ValueError):
        executor.step(np.zeros(6), np.zeros(6), phase="PAUSED")


def test_gripper_send_decision_uses_change_or_keepalive():
    assert gripper_send_decision(0.02, 0.019, 10.0, 10.1) == (True, False)
    assert gripper_send_decision(0.0201, 0.02, 10.0, 10.1) == (False, False)
    assert gripper_send_decision(0.02, 0.02, 10.0, 15.0) == (True, True)
