"""Re-execute native actions into a masked dataset and aligned real RGB videos.

Diagnostic exports remain ineligible. Formal exports require a pre-existing
frozen paired-scene protocol. No saved intermediate simulator state is restored.
"""
from pathlib import Path
import hashlib
import json
import mujoco
import numpy as np
import zarr
from numcodecs import Blosc
from . import VERSION
from .success_policy import required_success_steps, success_policy, policy_matches, validate_success_sequence
from .audit import digest
from .dataset import public_observation, pack_action, schema_metadata
from .environments import make_env
from .cameras import rigid_hand_geoms,obstruction_fraction,camera_quality,segmentation_renderer
from .protocol import load_frozen, scene_signature
from .provenance import capture, assert_current, code_manifest, model_manifest, file_hash

CAMERAS=('front','ego_left','ego_right','wrist_right','wrist_left')
INSTRUCTIONS={
    'hammer_nail':'Pick up the hammer and hammer the nail into the wooden block.',
    'pick_bucket':'Put the boxed object into the basket and lift the basket by its handle.',
    'bimanual_photograph':'Hold and aim the camera at the target, then press its shutter.',
    'bimanual_microwave_cook':'Put the food in the microwave, close its door, and press start.',
    'bimanual_unlock_ipad':'First lift the tablet off its stand with the left hand, orient and hold it steadily, then use the right hand to press 1, 2, then 3 while maintaining the left-hand grasp.',
    'single_hand_hanoi':'Move the three disks to the target post using the right hand, largest at the bottom, then release the stack.'
}
TRUTH_KEYS=('qpos','qvel','ctrl','mocap_pos','mocap_quat','time')


def camera_mapping(env):
    # Legacy dual output is left-wrist THEN right-wrist; canonical format is
    # always right THEN left, like state/action. Missing left wrist is masked.
    ids=list(env.camera_id)
    order=[0,1,2,4,3] if len(ids)==5 else [0,1,2,3]
    return {CAMERAS[i]:ids[j] for i,j in enumerate(order)}


def _compare(env, saved, errors, label):
    for key in TRUTH_KEYS:
        actual=np.asarray(getattr(env._data,key));expected=np.asarray(saved[key])
        if actual.shape!=expected.shape or not np.isfinite(expected).all():
            raise ValueError(f'Invalid {key} reference at {label}')
        error=float(np.max(np.abs(actual-expected),initial=0))
        errors[key]=max(errors[key],error)
    if max(errors.values())>1e-8:raise RuntimeError(f'Native action replay diverged at {label}: {errors}')


def export_episode(source,output,protocol_path=None,size=256):
    import imageio.v2 as imageio
    import cv2
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    meta=json.loads((source/'summary.json').read_text())
    task={'hammer':'hammer_nail','bucket':'pick_bucket'}.get(meta['task'],meta['task'])
    required=required_success_steps(task)
    if not meta.get('success') or meta.get('stable_success_steps',0)<required:
        raise ValueError(f'Source did not meet task success requirement: {required} step(s)')
    if not policy_matches(task,meta):
        raise ValueError('Bucket source predates first-success policy; physically rerun it')
    if not (source/'final_truth.json').is_file():raise ValueError('Missing terminal simulator reference')
    final=json.loads((source/'final_truth.json').read_text())
    protocol=load_frozen(protocol_path) if protocol_path else None
    if protocol and (size!=protocol['camera_size'] or task not in protocol['tasks']):
        raise ValueError('Task/image size disagrees with frozen protocol')
    if size<224 or size>640:raise ValueError('Expected VLA-ready real RGB resolution between 224 and 640')
    motion=zarr.open(str(source/'replay.zarr'),mode='r')['data']
    truth=zarr.open(str(source/'privileged.zarr'),mode='r')['data']
    n=len(motion['action'])
    if n<required or any(len(truth[k])!=n for k in TRUTH_KEYS+('success',)):
        raise ValueError('Unaligned trajectory/reference arrays')
    if not np.allclose(motion['timestamp'][:],np.arange(n)*.05,atol=1e-12,rtol=0):
        raise ValueError('Invalid 20 Hz source timestamps')
    env=make_env(task,meta['hand'],meta['seed']);renderer=None;seg_renderer=None;writers={}
    errors={key:0. for key in TRUTH_KEYS};stable=0;videos={};report={}
    try:
        if meta.get('model_sha256')!=env.hand_metadata['xml_sha256']:
            raise ValueError('Pilot model is not the current version; rerun the physical pilot')
        initial_scene=scene_signature(env)
        scene=None
        if protocol:
            scene=next((s for s in protocol['scenes'] if s['seed']==meta['seed']),None)
            if scene is None:raise ValueError('Pilot scene is not in the predeclared scene bank')
            if protocol['paired_scene_sha256'][task][scene['scene_id']]!=initial_scene:
                raise ValueError('Initial common scene differs from frozen scene')
            if protocol['models'][task+'/'+meta['hand']]!=model_manifest(task,meta['hand']):
                raise ValueError('Frozen model/assets changed')
        # This execution has fresh provenance. Earlier candidate labels are
        # never sufficient: every physical state and success flag is checked.
        provenance=capture(task,meta['hand'],source)
        output.mkdir(parents=True)
        (output/'videos').mkdir()
        store=zarr.open_group(str(output/'dataset.zarr'),mode='w')
        data=store.create_group('data');terminal=store.create_group('terminal')
        codec=Blosc(cname='zstd',clevel=3,shuffle=Blosc.BITSHUFFLE)
        def array(name,shape,dtype,chunks=None):
            return data.create_dataset(name,shape=(n,*shape),dtype=dtype,
                chunks=chunks or (min(n,64),*shape),compressor=codec)
        arrays={k:array(k,v.shape,v.dtype) for k,v in
                (public_observation(env)|pack_action(env,env.hold_action())).items()}
        images=array('images',(5,size,size,3),'u1',chunks=(1,1,size,size,3))
        image_mask=array('image_mask',(5,),'bool')
        timestamps=array('timestamp',(),'f8');success_array=array('success',(),'bool')
        terminated_array=array('terminated',(),'bool');truncated_array=array('truncated',(),'bool')
        native=store.create_group('replay_private')
        private={k:native.create_dataset(k,shape=(n,*np.asarray(getattr(env._data,k)).shape),dtype='f8',
            chunks=(min(n,64),*np.asarray(getattr(env._data,k)).shape),compressor=codec) for k in TRUTH_KEYS}
        camera_ids=camera_mapping(env);mask=np.array([name in camera_ids for name in CAMERAS])
        rigid_geoms=rigid_hand_geoms(env._model)
        obstruction_samples={name:[] for name in camera_ids if name.startswith('wrist_')}
        renderer=mujoco.Renderer(env._model,height=size,width=size)
        seg_renderer=segmentation_renderer(env._model,height=size,width=size)
        raw_hashes={name:hashlib.sha256() for name in camera_ids}
        for name in camera_ids:
            writers[name]=imageio.get_writer(str(output/'videos'/f'{name}.mp4'),fps=20,codec='libx264',
                quality=8,macro_block_size=None,ffmpeg_params=['-threads','1'])
        for i,command in enumerate(motion['action']):
            _compare(env,{k:truth[k][i] for k in TRUTH_KEYS},errors,f'pre-action {i}')
            command=np.asarray(command,dtype=np.float64)
            if task.startswith('bimanual'):
                width=6+len(env._native_ctrl_ids['right'])
                action={'right':command[:width],'left':command[width:]}
            else:action=command
            row=public_observation(env)|pack_action(env,action)
            for k,value in row.items():arrays[k][i]=value
            for k in TRUTH_KEYS:private[k][i]=getattr(env._data,k)
            image_mask[i]=mask;timestamps[i]=i*.05
            for channel,name in enumerate(CAMERAS):
                if name not in camera_ids:continue # zero fill, explicitly invalid camera
                renderer.update_scene(env._data,camera=camera_ids[name])
                rgb=renderer.render().copy()
                if rgb.shape!=(size,size,3) or rgb.dtype!=np.uint8 or rgb.std()<1:
                    raise ValueError('Missing/dummy/invalid real camera frame')
                images[i,channel]=rgb;writers[name].append_data(rgb);raw_hashes[name].update(rgb.tobytes())
                if name in obstruction_samples and (i%20==0 or i==n-1):
                    seg_renderer.update_scene(env._data,camera=camera_ids[name])
                    obstruction_samples[name].append(obstruction_fraction(seg_renderer,rigid_geoms))
            _,_,terminated,truncated,info=env.step(action)
            if any(w.number for w in env._data.warning):raise RuntimeError('MuJoCo numerical warning')
            success=bool(info.get('succeed',False));stable=stable+1 if success else 0
            if success!=bool(truth['success'][i]):raise RuntimeError(f'Task success sequence differs at step {i}')
            if task=='pick_bucket' and success and i!=n-1:
                raise RuntimeError('Bucket trajectory continues after first native success')
            if (terminated or truncated) and not success and i!=n-1:
                raise RuntimeError('Trajectory continues after an unsuccessful terminal condition')
            success_array[i]=success;terminated_array[i]=terminated;truncated_array[i]=truncated
        _compare(env,final,errors,'terminal')
        first_success=validate_success_sequence(task,success_array[:])
        quality=camera_quality(obstruction_samples)
        quality['segmentation_offscreen_samples']=0
        quality['rgb_offscreen_samples']=int(env._model.vis.quality.offsamples)
        for k,v in public_observation(env).items():terminal.create_dataset(k,data=v)
        for k in TRUTH_KEYS:terminal.create_dataset('sim_'+k,data=np.asarray(getattr(env._data,k)))
        for writer in writers.values():writer.close()
        writers.clear()
        # Independently decode EVERY compressed video frame. Video is lossy;
        # raw RGB is also stored losslessly in zarr and matched before encoding.
        for channel,name in enumerate(CAMERAS):
            if name not in camera_ids:continue
            path=output/'videos'/f'{name}.mp4';cap=cv2.VideoCapture(str(path))
            frames=0;max_mae=0.;rehash=hashlib.sha256()
            while True:
                ok,bgr=cap.read()
                if not ok:break
                if frames>=n or bgr.shape!=(size,size,3):raise ValueError('Wrong decoded video shape/length')
                rgb=images[frames,channel];rehash.update(rgb.tobytes())
                mae=float(np.mean(np.abs(bgr[...,::-1].astype(float)-rgb.astype(float))))
                max_mae=max(max_mae,mae);frames+=1
            fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
            if frames!=n or abs(fps-20)>1e-6 or max_mae>12 or rehash.hexdigest()!=raw_hashes[name].hexdigest():
                raise ValueError(f'Video/RGB alignment or encoding failed: {name}, {frames}, {fps}, {max_mae}')
            videos[name]=dict(frames=frames,fps=fps,shape=[size,size,3],max_compression_mae=max_mae,
                              raw_rgb_sha256=rehash.hexdigest(),sha256=file_hash(path))
        assert_current(provenance,task,meta['hand'],source)
        if protocol:load_frozen(protocol_path)
        schema=schema_metadata(env)
        schema.update(image_channels=list(CAMERAS),image_shape=[5,size,size,3],
            image_padding='only missing left wrist camera is zero-filled and masked for single-arm tasks',
            instruction=INSTRUCTIONS[task],observation_timing='pre-action; success/termination are post-action',
            action_dtype='float64 for exact replay; trainer may explicitly cast after loading',
            terminal_tail=('ends at first native success, no additional hold' if success_policy(task)['terminate_on_first_success'] else
                           'up to 10 stable-success control steps, including underlying success termination signals'))
        metadata=dict(version=VERSION,hand=meta['hand'],task=task,seed=meta['seed'],steps=n,
            success_policy=success_policy(task),first_success_step=first_success,
            schema=schema,source_candidate=str(source),provenance=provenance,
            source_generator_provenance=meta.get('provenance'),
            protocol_sha256=protocol['sha256'] if protocol else None,scene=scene,
            initial_scene_sha256=initial_scene,camera_quality=quality,training_eligible=bool(protocol),
            release_scope=protocol.get('scope','suite') if protocol else 'diagnostic',
            scope='formal protocol episode' if protocol else 'diagnostic pipeline validation only')
        store.attrs.update(metadata)
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        files={str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
        report=dict(passed=True,version=VERSION,task=task,hand=meta['hand'],seed=meta['seed'],steps=n,
            stable_success_steps=stable,success_policy=success_policy(task),first_success_step=first_success,
            training_eligible=bool(protocol),rgb_verified=True,
            camera_quality=quality,
            action_replay_max_error=errors,videos=videos,files=files,
            code_sha256=digest(provenance['code']),model_manifest_sha256=provenance['model']['sha256'],
            protocol_sha256=metadata['protocol_sha256'],initial_scene_sha256=initial_scene)
    except Exception as exc:
        report=dict(passed=False,training_eligible=False,error=repr(exc),action_replay_max_error=errors)
        raise
    finally:
        for writer in writers.values():writer.close()
        if renderer is not None:renderer.close()
        if seg_renderer is not None:seg_renderer.close()
        env.close()
        if output.exists():(output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
