import pickle, tempfile, unittest, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
from episode_tools import JOINT_KEYS, commit_episode, load_config, suggest_trim_range, validate_episode
def frame(i, moving=False):
 vals=[float(i) if moving else 0.0]*6+[.02]
 obs={k:v for k,v in zip(JOINT_KEYS,vals)}; act=dict(obs)
 obs.update(top=np.zeros((3,4,3),np.uint8),wrist=np.zeros((3,4,3),np.uint8))
 return {'observation':obs,'action':act,'timestamp':i/30}
class Tests(unittest.TestCase):
 def make(self,root,n=10):
  ep=root/'episode_000001_temp'; ep.mkdir()
  for i in range(n):
   with (ep/f'frame_{i:06d}.pkl').open('wb') as f: pickle.dump(frame(i,2<=i<=6),f)
  return ep
 def test_trim_commit_and_manifest(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); ep=self.make(root); cfg=load_config(); s=suggest_trim_range(__import__('episode_tools').load_raw_episode(ep),cfg)
   final=commit_episode(ep,root,0,8,cfg,delete_source=False,auto=s)
   self.assertTrue((final/'review_manifest.json').exists()); self.assertTrue(ep.exists()); self.assertTrue(validate_episode(final)['valid'])
 def test_bad_timestamp(self):
  with tempfile.TemporaryDirectory() as d:
   ep=self.make(Path(d)); p=ep/'frame_000001.pkl'
   with p.open('rb') as f: x=pickle.load(f)
   x['timestamp']=0
   with p.open('wb') as f: pickle.dump(x,f)
   self.assertFalse(validate_episode(ep)['valid'])
 def test_commit_failure_keeps_source_and_no_final(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); ep=self.make(root); cfg=load_config()
   # An invalid range fails before any final directory is published.
   with self.assertRaises(ValueError): commit_episode(ep,root,0,99,cfg,delete_source=True)
   self.assertTrue(ep.exists()); self.assertFalse((root/'episode_000001').exists())
if __name__=='__main__': unittest.main()
