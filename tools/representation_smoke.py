#!/usr/bin/env python3
"""Exercise Native, FAAS, and DexLatent through the public runtime."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation'))
from runtime import Runtime

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--representations',nargs='+',choices=('native','faas','dexlatent'),default=['native','faas'])
 args=parser.parse_args()
 reports=[]
 for rep in dict.fromkeys(args.representations):
  runtime=Runtime('bimanual_microwave_cook','xhand',0,'grouped80_v1',rep,render=False)
  try:
   hold=runtime.e.hold_action(); encoded=runtime.encode_action(hold)
   decoded,_=runtime.decode_action(encoded,clip=False)
   source=hold if isinstance(hold,dict) else {'right':hold}
   error=max(float(np.max(np.abs(decoded[s]-source[s]))) for s in source)
   assert np.isfinite(encoded).all() and np.isfinite(error)
   if rep in ('native','faas'): assert error<1e-8
   runtime.step(encoded)
   reports.append({'representation':rep,'passed':True,'action_roundtrip_max_abs':error,'physics_step_finite':bool(np.isfinite(runtime.d.qpos).all())})
  finally: runtime.close()
 result={'passed':len(reports)==len(set(args.representations)),'task':'bimanual_microwave_cook','hand':'xhand','representations':reports}
 (ROOT/'REPRESENTATION_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
