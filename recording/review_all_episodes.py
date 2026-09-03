#!/usr/bin/env python3
"""Sequential review to a separate output root; inputs remain read-only."""
import argparse, json
from pathlib import Path
from review_episode import review
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input-root',required=True); ap.add_argument('--output-root',required=True); args=ap.parse_args()
 root=Path(args.output_root)
 if root.exists():
  if any(root.iterdir()): raise SystemExit(f'output root already exists and is not empty: {root}')
 else: root.mkdir(parents=True)
 episodes=[p for p in sorted(Path(args.input_root).glob('episode_*')) if p.is_dir() and not p.name.endswith(('_temp','_commit_tmp'))]
 results=[]; index=0
 while 0 <= index < len(episodes):
  ep=episodes[index]; print(f'Episode {index+1} / {len(episodes)}: {ep.name}')
  result=review(ep,root,False)
  results.append({'episode':ep.name,'status':result})
  (root/'review_session.json').write_text(json.dumps({'input_root':str(args.input_root),'results':results},ensure_ascii=False,indent=2))
  if result=='stop': break
  if result=='previous': index=max(0,index-1); continue
  index+=1
if __name__=='__main__': main()
