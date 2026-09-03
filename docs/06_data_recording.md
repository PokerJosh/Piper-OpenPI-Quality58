# 06 — Data Recording

The final recording path is `recording/custom_record_final.py`; the simpler `custom_record.py` and `custom_record_v2.py` are retained as historical variants. The launcher is `recording/record_piper_teleop.sh`.

## Episode procedure

1. Complete leader and follower calibration.
2. Independently verify CAN and camera readiness.
3. Start the approved leader/follower teleoperation process.
4. Start an episode only after the scene and robot are in the demonstration start pose.
5. Capture synchronized top/wrist images, seven-dimensional observation state, seven-dimensional action, timestamps, and task prompt.
6. Perform the task with the leader; stop at the final task state.
7. Stop the episode, review it, and either approve or discard it.
8. Reset the scene and robot **outside recording**.
9. Start the next episode only after the reset is complete and stable.

The guarded recorder reserves a temporary episode directory, validates frames, supports automatic trim plus pre/post roll, and atomically commits approved episodes. `g` approves; `b` discards a temporary failed attempt; `s` stops without silently accepting.

Do not record reset motion in a demonstration: it adds actions unrelated to the task, changes the state/action distribution, and can teach the policy to undo a successful placement. Bad episodes remain separate until review; do not overwrite an existing episode index.

Raw episodes, images, and videos are intentionally absent from this archive.
