#!/usr/bin/env python3
"""Visual replay and guarded, single-episode Piper replay preflight."""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
import numpy as np
from episode_tools import CAMERA_KEYS, JOINT_KEYS, list_episode_frames, load_config, load_raw_episode, validate_episode

# Piper plugin limits in degrees; kept here for offline preflight so dry-run never
# imports or connects the hardware layer.
PIPER_JOINT_LIMITS_DEG = ((-150.0, 124.2), (0.0, 179.9), (-170.0, 0.0), (-100.0, 100.0), (-69.9, 69.9), (-120.0, 120.0))

def _draw(frame, index, total, selected=None):
    import cv2
    obs, act = frame['observation'], frame['action']
    images = [obs[k].copy() for k in CAMERA_KEYS]
    text = f"Frame {index}/{total-1}  t={float(frame['timestamp']):.3f}s"
    values = "obs " + ' '.join(f"{k[:2]}={float(obs[k]):.3f}" for k in JOINT_KEYS) + " | act " + ' '.join(f"{float(act[k]):.3f}" for k in JOINT_KEYS)
    for image, label in zip(images, ('TOP', 'WRIST'), strict=True):
        cv2.putText(image, label + '  ' + text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 1)
        cv2.putText(image, values, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .38, (0,255,0), 1)
    return np.hstack(images)

def export_video(frames, output: Path, speed=1.0):
    import cv2
    first = _draw(frames[0], 0, len(frames)); h, w = first.shape[:2]
    fps = 30 * speed
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    if not writer.isOpened(): raise RuntimeError('VideoWriter mp4v unavailable; choose a writable .avi path')
    for i, frame in enumerate(frames): writer.write(_draw(frame, i, len(frames)))
    writer.release()

def visual(frames, speed, export):
    if export: export_video(frames, Path(export), speed); print(f'VIDEO_EXPORTED={export}')
    import cv2
    i, playing = 0, True
    while True:
        cv2.imshow('Piper visual replay [space pause, j/l step, q exit]', _draw(frames[i], i, len(frames)))
        key = cv2.waitKey(max(1, int(1000/(30*speed))) if playing else 0) & 0xff
        if key == ord('q'): break
        if key == ord(' '): playing = not playing
        if key == ord('j'): i=max(0,i-1); playing=False
        elif key == ord('l'): i=min(len(frames)-1,i+1); playing=False
        elif playing: i=(i+1) % len(frames)
    cv2.destroyAllWindows()

def preflight(frames, config):
    result=validate_episode_from_frames(frames, config)
    deltas=np.abs(np.diff(np.array([[f['action'][k] for k in JOINT_KEYS[:6]] for f in frames],dtype=float),axis=0)) if len(frames)>1 else np.empty((0,6))
    mx=float(deltas.max()) if deltas.size else 0.0
    result['max_joint_action_delta_deg']=mx
    result['REAL_REPLAY_SAFE_UNDER_CURRENT_GATE']=mx <= config['replay']['max_joint_step_deg']
    return result

def validate_episode_from_frames(frames, config):
    # On-disk validator remains authoritative; this adds physical action checks for dry run.
    errors=[]
    for i,f in enumerate(frames):
        values=np.asarray([f['action'][k] for k in JOINT_KEYS],dtype=float)
        if not np.isfinite(values).all(): errors.append(f'frame {i}: nonfinite action')
        if not config['replay']['gripper_min_m'] <= values[6] <= config['replay']['gripper_max_m']: errors.append(f'frame {i}: gripper outside replay range')
        for joint, (value, limits) in enumerate(zip(values[:6], PIPER_JOINT_LIMITS_DEG, strict=True), 1):
            if not limits[0] <= value <= limits[1]: errors.append(f'frame {i}: J{joint} outside Piper joint limits')
    return {'DATA_VALID': not errors, 'errors':errors}

def real_replay(episode: Path, frames, config, speed: float, max_start_pose_error_deg: float):
    # Reached only by explicit ACK + --real.  No batch/glob accepted by argparse path validation.
    from dataclasses import replace
    import draccus, yaml
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot.robots.utils import make_robot_from_config
    register_third_party_plugins()
    cfg_path=Path(__file__).with_name('piper_camera_config.yaml')
    cfg=draccus.decode(RecordConfig, yaml.safe_load(cfg_path.read_text()))
    # Runtime-only safety overrides: no cameras and never home on disconnect.
    robot_cfg=replace(cfg.robot, cameras={}, enable_motion=True, go_home_on_disconnect=False)
    robot=make_robot_from_config(robot_cfg); robot.connect()
    try:
        current=robot.get_observation(); recorded=frames[0]['observation']
        err=max(abs(float(current[k])-float(recorded[k])) for k in JOINT_KEYS[:6])
        print(f'CURRENT={[current[k] for k in JOINT_KEYS[:6]]}\nRECORDED_START={[recorded[k] for k in JOINT_KEYS[:6]]}\nSTART_ERROR={err}')
        if err > max_start_pose_error_deg: raise RuntimeError('start pose safety gate failed')
        for i, frame in enumerate(frames):
            requested=frame['action']; effective=robot.send_action(requested); feedback=robot.get_observation()
            tracking={k: float(feedback[k])-float(effective[k]) for k in JOINT_KEYS}
            print({'frame':i, 'requested_action':requested, 'effective_action':effective, 'feedback':feedback, 'tracking_error_vs_effective':tracking})
            time.sleep(1 / (config['fps'] * speed))
    finally:
        robot.disconnect()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episode',required=True); ap.add_argument('--mode',choices=('visual','robot'),default='visual'); ap.add_argument('--speed',type=float,default=1.0); ap.add_argument('--export-video'); ap.add_argument('--real',action='store_true'); ap.add_argument('--max-start-pose-error-deg',type=float,help='One-run real replay override; default comes from record_review_config.yaml.'); args=ap.parse_args()
    episode=Path(args.episode)
    if not episode.is_dir() or any(x in args.episode for x in '*?['): ap.error('--episode must be one explicit episode directory')
    cfg=load_config(); frames=load_raw_episode(episode)
    if not frames: raise SystemExit('empty episode')
    if args.mode=='visual': visual(frames,args.speed,args.export_video); return
    print('REPLAY_DRY_RUN\nPIPER_CONNECTED = NO\nMOTION = DISABLED\nCURRENT_STEP = Offline trajectory validation\nNEXT_ACTION = Review DATA_VALID and REAL_REPLAY_SAFE_UNDER_CURRENT_GATE.')
    disk_validation=validate_episode(episode,cfg['fps'])
    report=preflight(frames,cfg)
    report['DATA_VALID']=bool(report['DATA_VALID'] and disk_validation['valid'])
    report['dataset_validation_errors']=disk_validation['errors']
    print(report)
    if not args.real or os.environ.get('PIPER_REPLAY_ACK') != 'I_UNDERSTAND':
        print('REAL_REPLAY_GATE=BLOCKED\nNEXT_ACTION = Motion remains disabled. Real replay requires one explicit episode, --real, and PIPER_REPLAY_ACK=I_UNDERSTAND.'); return
    if not report['DATA_VALID'] or not report['REAL_REPLAY_SAFE_UNDER_CURRENT_GATE']: raise SystemExit('real replay preflight failed')
    print('REAL_REPLAY_AUTHORIZED = YES\nWARNING = This command will connect to Piper and replay one recorded trajectory.\nSAFETY = It will not auto-home; start pose will be checked before sending any action.\nCURRENT_STEP = Connecting only after all gates passed.')
    start_gate = args.max_start_pose_error_deg if args.max_start_pose_error_deg is not None else cfg['replay']['max_start_pose_error_deg']
    print(f'START_POSE_ERROR_GATE_DEG = {start_gate}')
    real_replay(episode,frames,cfg,args.speed,start_gate)
if __name__=='__main__': main()
