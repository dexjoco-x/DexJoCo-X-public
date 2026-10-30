#!/usr/bin/env python3
"""Strict replay of saved Native actions, independently of codec acceptance.

Uses the public Native helpers unchanged. Success must occur within the saved
sequence, before termination; no appended controls or intermediate state resets.
By default every episode in bundle/manifests is checked, including train entries.
"""
import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from audit_dexlatent_validation_replay import initial_error, commands, step
from episode_validation import load_validated_episode


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def environment_fingerprint():
    import mujoco
    import numpy.core._multiarray_umath as numpy_extension
    import scipy.spatial.transform._rotation as scipy_rotation
    simulator = str(Path(__file__).resolve().parents[1] / 'simulator')
    if simulator not in sys.path:
        sys.path.insert(0, simulator)
    from dexjoco import numerics
    from numpy.core._multiarray_umath import __cpu_features__
    packages={}
    for name in ('mujoco','numpy','scipy','gymnasium','dm-robotics-transformations'):
        packages[name]=importlib.metadata.version(name)
    libraries={}
    system_libraries={}
    maps=Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            path=Path(line.split()[-1])
            if 'libmujoco.so' in path.name and path.is_file():
                libraries[path.name]=sha256(path)
            if path.is_file() and (path.name.startswith(('libm-', 'libc-')) or path.name in ('libm.so.6','libc.so.6')):
                system_libraries[path.name]=sha256(path)
    try:
        from threadpoolctl import threadpool_info
        blas=[{k:r.get(k) for k in ('internal_api','version','threading_layer','architecture','num_threads')}
              for r in threadpool_info() if r.get('user_api')=='blas']
    except ImportError:
        blas=None
    return {'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
            'numpy_cpu_features':{k:bool(v) for k,v in __cpu_features__.items()},
            'numpy_disabled_cpu_features':os.environ.get('NPY_DISABLE_CPU_FEATURES'),
            'system_libraries_sha256':system_libraries,
            'controller_math':numerics.math_fingerprint(),
            'controller_math_sha256':sha256(numerics.__file__),
            'packages':packages,'mujoco_version':mujoco.mj_versionString(),
            'mujoco_library_sha256':libraries,'blas':blas,
            'numpy_extension_sha256':sha256(numpy_extension.__file__),
            'scipy_rotation_extension_sha256':sha256(scipy_rotation.__file__),
            'native_execution_helpers_sha256':{fn.__name__:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                                              for fn in (initial_error,commands,step)},
            'numerical_environment':{k:os.environ.get(k) for k in
               ('OPENBLAS_CORETYPE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}


def replay(job):
    from runtime import Runtime, CONTRACTS
    bundle,row=Path(job[0]),job[1]
    check_states=bool(job[2]) if len(job)>2 else False
    result={k:row[k] for k in ('task','hand','scene_id','seed','tensor','tensor_sha256')}
    runtime=None
    try:
        states,actions=load_validated_episode(bundle,row,CONTRACTS[row['task']+'/'+row['hand']])
        result['data_contract_and_arrays_verified']=True
        result['saved_action_dtype']=str(actions.dtype)
        runtime=Runtime(row['task'],row['hand'],row['seed'],'arm40_v1',representation='native',render=False)
        result['model_binding']=runtime.model_binding
        error=initial_error(runtime,states[0]);result['initial_robot_error']=error
        if error>2e-5:
            raise ValueError('Initial robot state does not match the recorded state')
        first=None
        state_error=0.0
        for index,action in enumerate(actions):
            if check_states:
                import mujoco
                # Derive FK on render data, leaving all physics caches intact.
                for key in ('qpos','qvel','act','ctrl','mocap_pos','mocap_quat','qacc_warmstart'):
                    getattr(runtime.shadow,key)[:]=getattr(runtime.d,key)
                runtime.shadow.time=runtime.d.time
                mujoco.mj_forward(runtime.m,runtime.shadow)
                from scipy.spatial.transform import Rotation
                for arm,side in enumerate(('right','left')):
                    if side not in runtime.e._native_ctrl_ids:
                        continue
                    site=runtime.e._tcp_sites[side]
                    q=runtime.shadow.qpos[runtime.e._full_hand_qpos[side]]
                    expected_rotation=Rotation.from_quat(states[index,arm,3:7],scalar_first=True).as_matrix()
                    for delta in (runtime.shadow.site_xpos[site]-states[index,arm,:3],
                                  runtime.shadow.site_xmat[site].reshape(3,3)-expected_rotation,
                                  q-states[index,arm,7:7+len(q)]):
                        if not np.isfinite(delta).all():
                            raise ValueError('Non-finite state during recorded-state verification')
                        state_error=max(state_error,float(np.max(np.abs(delta))))
            info,terminated,truncated=step(runtime,commands(runtime,action))
            if info.get('succeed',False) and first is None:
                first=index
            if terminated or truncated:
                break
        result.update(passed=first is not None,first_success_step=first,
                      saved_steps=len(actions),executed_steps=index+1,
                      terminated=terminated,truncated=truncated,
                      stop_reason=('terminated' if terminated else 'truncated' if truncated else 'actions_exhausted'),
                      final_info={k:v.item() if isinstance(v,np.generic) else v for k,v in info.items()
                                  if isinstance(v,(bool,int,float,str,np.generic))},error=None)
        if index+1!=len(actions):
            raise ValueError('Saved sequence contains actions after environment termination')
        if check_states:
            result.update(max_recorded_robot_state_error=state_error,
                          recorded_state_fidelity_passed=state_error<=2e-5)
            result['task_success']=result['passed']
            result['passed']=result['passed'] and result['recorded_state_fidelity_passed']
    except Exception as error:
        result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        if runtime is not None:
            runtime.close()
            runtime=None
        gc.collect()
        trim=getattr(ctypes.CDLL(None),'malloc_trim',None)
        if trim is not None:
            trim(0)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('release',type=Path,help='Directory containing bundle/, or bundle/ itself')
    p.add_argument('--task',action='append')
    p.add_argument('--hand',action='append')
    p.add_argument('--scene',action='append')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--check-states',action='store_true',help='Also compare each recorded pre-action robot state using separate FK data')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    bundle=args.release if (args.release/'manifests').is_dir() else args.release/'bundle'
    manifests=sorted((bundle/'manifests').glob('*.json'))
    if not manifests:
        p.error('No bundle/manifests/*.json files found')
    if args.workers<1:
        p.error('--workers must be positive')
    rows=[r for path in manifests for r in json.loads(path.read_text())
          if (not args.task or r['task'] in args.task)
          and (not args.hand or r['hand'] in args.hand)
          and (not args.scene or r['scene_id'] in args.scene)]
    if not rows:
        p.error('No episodes match the requested filters')
    keys=[(r['task'],r['hand'],r['scene_id']) for r in rows]
    if len(set(keys))!=len(keys):
        p.error('Duplicate episode keys in manifests')
    progress=args.output.with_suffix('.episodes.jsonl')
    if args.output.exists() or progress.exists():
        p.error('Output already exists; choose a new path to preserve audit records')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    env=environment_fingerprint()
    records=[]
    with progress.open('x') as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(replay,(str(bundle),row,args.check_states)) for row in rows]
        for future in as_completed(futures):
            record=future.result();records.append(record)
            stream.write(json.dumps(record)+'\n');stream.flush()
            print(f"{len(records)}/{len(rows)} {record['task']}/{record['hand']}/{record['scene_id']} passed={record['passed']}",flush=True)
    records.sort(key=lambda r:(r['task'],r['hand'],r['scene_id']))
    succeeded=sum(r['passed'] for r in records)
    report={'passed':succeeded==len(rows),'episodes':len(rows),'successes':succeeded,
            'failures':len(rows)-succeeded,'execution_errors':sum(r['error'] is not None for r in records),
            'scope':'all manifest splits; explicit task/hand/scene filters only',
            'filters':{'task':args.task,'hand':args.hand,'scene':args.scene},
            'state_fidelity_checked':args.check_states,
            'manifest_sha256':{path.name:sha256(path) for path in manifests},
            'environment':env,'records':records}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','environment','manifest_sha256')},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
