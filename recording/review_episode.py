#!/usr/bin/env python3
"""Human review UI. It never creates a robot/CAN connection."""
from __future__ import annotations
import argparse
from pathlib import Path
from episode_tools import commit_episode, discard_episode, load_config, load_raw_episode, suggest_trim_range
from replay_episode import _draw

def review(episode, output_root=None, allow_discard=False):
    import cv2
    cfg=load_config(); frames=load_raw_episode(episode); auto=suggest_trim_range(frames,cfg); start,end=auto['start'],auto['end']; i=start; playing=False
    while True:
        canvas=_draw(frames[i],i,len(frames)); h,w=canvas.shape[:2]
        line=f'AUTO {auto["start"]}:{auto["end"]} | IN {start} CUR {i} OUT {end} | Keep {end-start+1} | SPACE play j/l +/-1 J/L +/-30 i/o mark a auto r full v preview g approve b reject n skip p previous q stop'
        cv2.putText(canvas,line,(10,h-14),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1)
        cv2.imshow('Piper review (NO ROBOT / NO CAN)',canvas)
        k=cv2.waitKey(33 if playing else 0) & 0xff
        if k==ord('q'):
            cv2.destroyAllWindows(); return 'stop'
        if k==ord('n'):
            cv2.destroyAllWindows(); return 'skipped'
        if k==ord('p'):
            cv2.destroyAllWindows(); return 'previous'
        if k==ord(' '): playing=not playing
        if k==ord('j'): i=max(0,i-1); playing=False
        elif k==ord('l'): i=min(len(frames)-1,i+1); playing=False
        elif k==ord('J'): i=max(0,i-30); playing=False
        elif k==ord('L'): i=min(len(frames)-1,i+30); playing=False
        elif k==ord('i'): start=min(i,end)
        elif k==ord('o'): end=max(i,start)
        elif k==ord('a'): start,end=auto['start'],auto['end']
        elif k==ord('r'): start,end=0,len(frames)-1
        elif k==ord('v'): i=start; playing=True
        elif k==ord('g'):
            target=Path(output_root) if output_root else Path(episode).parent
            final=commit_episode(episode,target,start,end,cfg,delete_source=(output_root is None),auto=auto); print(f'APPROVED={final}'); cv2.destroyAllWindows(); return 'approved'
        elif k==ord('b'):
            if allow_discard: discard_episode(episode); print('DISCARDED_TEMP')
            else: print('REJECTED: input episode kept unchanged')
            cv2.destroyAllWindows(); return 'rejected'
        elif playing: i=min(end,i+1); playing=i<end
    cv2.destroyAllWindows(); return 'stop'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episode',required=True); ap.add_argument('--output-root'); args=ap.parse_args()
    ep=Path(args.episode); review(ep,args.output_root,ep.name.endswith('_temp') and args.output_root is None)
if __name__=='__main__': main()
