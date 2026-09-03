#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from episode_tools import list_episode_frames
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',default='dataset_review_report.json'); args=ap.parse_args()
 rows=[]
 for ep in sorted(Path(args.root).glob('episode_*')):
  m=ep/'review_manifest.json'
  if not m.exists(): continue
  data=json.loads(m.read_text()); data['episode']=ep.name; rows.append(data)
 summary={'TOTAL_EPISODES':len(rows),'APPROVED':sum(x.get('review_status')=='approved' for x in rows),'REJECTED':sum(x.get('review_status')=='rejected' for x in rows),'SKIPPED':0,
 'ORIGINAL_FRAMES':sum(x['original_frame_count'] for x in rows),'KEPT_FRAMES':sum(x['saved_frame_count'] for x in rows),'REMOVED_HEAD_FRAMES':sum(x['trim_start_original'] for x in rows),'REMOVED_TAIL_FRAMES':sum(x['original_frame_count']-1-x['trim_end_original'] for x in rows),'episodes':rows}
 summary['TOTAL_DURATION_BEFORE']=summary['ORIGINAL_FRAMES']/30; summary['TOTAL_DURATION_AFTER']=summary['KEPT_FRAMES']/30
 output=Path(args.output); output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
