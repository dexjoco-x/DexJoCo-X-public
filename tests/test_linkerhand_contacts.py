"""Physical regression: bounded drive, self-contact, obstruction, release."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,xml.etree.ElementTree as ET
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parent
ASSETS=ROOT.parent/'simulator/dexjoco/sim/envs/xmls'

def metrics(m,d,out):
    assert not any(d.warning.number), str(d.warning.number)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    assert np.all(m.actuator_forcelimited)
    assert np.all(d.actuator_force <= m.actuator_forcerange[:,1]+1e-10)
    assert np.all(d.actuator_force >= m.actuator_forcerange[:,0]-1e-10)
    if d.ncon:out['penetration_mm']=max(out['penetration_mm'],-float(min(d.contact.dist))*1000)
    out['linkage_rad']=max(out['linkage_rad'],float(max(abs(d.qpos[m.jnt_qposadr[m.eq_obj1id]]-m.eq_data[:,1]*d.qpos[m.jnt_qposadr[m.eq_obj2id]]))))
    out['velocity_rad_s']=max(out['velocity_rad_s'],float(max(abs(d.qvel))))

def run(job):
    hand,side=job;path=ASSETS/hand/side/'hand.xml';m=mujoco.MjModel.from_xml_path(str(path));rows=[]
    for mode in ['ramp','step','random123','random4']:
        d=mujoco.MjData(m);out=dict(hand=hand,side=side,mode=mode,penetration_mm=0,linkage_rad=0,velocity_rad_s=0,warnings=[])
        rng=np.random.default_rng(4 if mode=='random4' else 123);targets=rng.uniform(*m.actuator_ctrlrange.T,size=(10,m.nu))
        try:
            for i in range(5000):
                t=i*.002;amount=min(t/3,1,max(0,(9-t)/3))
                d.ctrl[:]=m.actuator_ctrlrange[:,1]*amount if mode=='ramp' else m.actuator_ctrlrange[:,1]*(i<3500) if mode=='step' else targets[min(i//500,9)]
                mujoco.mj_step(m,d);metrics(m,d,out)
                assert abs(d.time-(i+1)*.002)<1e-8
            out['passed']=out['penetration_mm']<1 and out['linkage_rad']<.02
        except Exception as exc:out.update(passed=False,error=str(exc))
        rows.append(out)
    # Obstruct one independently driven finger, using a rigid object placed in
    # its closing path. At the open posture the object must not intersect it.
    aid=next(i for i,j in enumerate(m.actuator_trnid[:,0]) if m.joint(j).name.endswith('index_mcp_pitch'))
    jid=m.actuator_trnid[aid,0];qid=m.jnt_qposadr[jid]
    bid=m.body(side[0]+'h_index_distal').id
    gid=next(g for g in range(m.ngeom) if m.geom_bodyid[g]==bid and m.geom_contype[g])
    obstacle=None
    for fraction in [.6,.75,.9,1.]:
        d=mujoco.MjData(m);d.qpos[qid]=m.actuator_ctrlrange[aid,1]*fraction
        for j1,j2,c in zip(m.eq_obj1id,m.eq_obj2id,m.eq_data):d.qpos[m.jnt_qposadr[j1]]=c[1]*d.qpos[m.jnt_qposadr[j2]]
        mujoco.mj_forward(m,d);pos=d.geom_xpos[gid].copy()
        spec=mujoco.MjSpec.from_file(str(path))
        spec.worldbody.add_geom(name='test_obstacle',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.012,0,0],pos=pos,rgba=[.2,.7,.9,1])
        # Test fixture holds the other active axes (including L30's wrist).
        # Only the tested finger and its passive linkage can move.
        for other in m.actuator_trnid[:,0]:
            if other!=jid:
                spec.add_equality(name='fixture_'+m.joint(other).name,type=mujoco.mjtEq.mjEQ_JOINT,
                                  name1=m.joint(other).name,data=[0]*11,solref=[.004,1],solimp=[.9999,.9999,.001,.5,2])
        block=spec.compile();bd=mujoco.MjData(block);mujoco.mj_forward(block,bd);oid=block.geom('test_obstacle').id
        if not any(c.dist<0 and oid in (c.geom1,c.geom2) for c in bd.contact):obstacle=block;break
    assert obstacle is not None,'could not place a nonintersecting obstacle'
    m=obstacle;d=mujoco.MjData(m);out=dict(hand=hand,side=side,mode='obstruction_release',penetration_mm=0,linkage_rad=0,velocity_rad_s=0,warnings=[],object_contacts=0,obstacle_position=pos.tolist())
    try:
        for i in range(7500):
            t=i*.002;amount=min(t/3,1,max(0,(13-t)/3));d.ctrl[:]=0;d.ctrl[aid]=m.actuator_ctrlrange[aid,1]*amount
            mujoco.mj_step(m,d);metrics(m,d,out)
            assert abs(d.time-(i+1)*.002)<1e-8
            out['object_contacts']+=sum(oid in (c.geom1,c.geom2) for c in d.contact)
            if i==4999:out['blocked_error_rad']=float(m.actuator_ctrlrange[aid,1]-d.qpos[qid])
        out['released_error_rad']=abs(float(d.qpos[qid]))
        out['passed']=out['penetration_mm']<1 and out['linkage_rad']<.02 and out['object_contacts']>0 and out['blocked_error_rad']>.03 and out['released_error_rad']<.03
    except Exception as exc:out.update(passed=False,error=str(exc))
    rows.append(out)
    return rows

import unittest
import yaml

class StandaloneLinkerHandContactTests(unittest.TestCase):
    def test_force_caps_and_servo_metadata(self):
        for hand in ['linkerhand_o6','linkerhand_l6','linkerhand_l30']:
            directory=ASSETS/hand
            metadata=yaml.safe_load((directory/'meta.yaml').read_text())
            for side in ['left','right']:
                with self.subTest(hand=hand,side=side):
                    model=mujoco.MjModel.from_xml_path(str(directory/side/'hand.xml'))
                    self.assertTrue(np.all(model.actuator_forcelimited))
                    self.assertTrue(np.all(model.actuator_forcerange[:,1]>0))
                    np.testing.assert_allclose(model.actuator_gainprm[:,0],metadata['default_gains']['stiffness'])
                    np.testing.assert_allclose(-model.actuator_biasprm[:,2],metadata['default_gains']['damping'])
                    joints={j.get('name'):j for j in ET.parse(directory/side/'source.urdf').getroot().findall('joint')}
                    for jid in range(model.njnt):
                        dynamics=joints[model.joint(jid).name].find('dynamics')
                        self.assertEqual(model.dof_damping[model.jnt_dofadr[jid]],float(dynamics.get('damping','0')) if dynamics is not None else 0)
                        self.assertEqual(model.dof_frictionloss[model.jnt_dofadr[jid]],float(dynamics.get('friction','0')) if dynamics is not None else 0)
                    for aid,jid in enumerate(model.actuator_trnid[:,0]):
                        name=model.joint(jid).name
                        effort=float(joints[name].find('limit').get('effort'))
                        if hand=='linkerhand_l30' and effort==0:
                            self.assertEqual(side,'left')
                            self.assertIn(name,('lh_thumb_cmc_roll','lh_thumb_mcp','lh_thumb_dip'))
                            right={j.get('name'):j for j in ET.parse(directory/'right/source.urdf').getroot().findall('joint')}
                            effort=float(right['rh_'+name[3:]].find('limit').get('effort'))
                        if hand=='linkerhand_o6':
                            # Preserve the original MJCF's tighter per-side caps;
                            # generic URDF effort=100 is not the operational cap.
                            caps=([1,4,2.55,2.55,2.55,2.55] if side=='left' else [1,3.8148,2.55,2.55,2.55,2.55])
                            effort=caps[aid]
                        np.testing.assert_allclose(model.actuator_forcerange[aid],[-effort,effort])

    def test_each_active_joint_reaches_and_releases(self):
        # Linkage-only tests omit independent thumb opposition and wrist axes.
        # Keep all self-collisions enabled to catch false blocking by CAD hulls.
        for hand in ['linkerhand_o6','linkerhand_l6','linkerhand_l30']:
            for side in ['left','right']:
                model=mujoco.MjModel.from_xml_path(str(ASSETS/hand/side/'hand.xml'))
                for aid,jid in enumerate(model.actuator_trnid[:,0]):
                    ends=[model.actuator_ctrlrange[aid,1]]
                    if model.actuator_ctrlrange[aid,0]<-.01:ends.append(model.actuator_ctrlrange[aid,0])
                    for end in ends:
                        with self.subTest(hand=hand,side=side,joint=model.joint(jid).name,end=end):
                            data=mujoco.MjData(model);target=.8*end;qid=model.jnt_qposadr[jid]
                            for i in range(4000):
                                data.ctrl[aid]=target*min(1,i/1500)
                                mujoco.mj_step(model,data)
                                self.assertFalse(any(data.warning.number))
                            self.assertLess(abs(data.qpos[qid]-target),.08,'active joint blocked during a free-space motion')
                            for i in range(4000):
                                data.ctrl[aid]=target*max(0,1-i/1500)
                                mujoco.mj_step(model,data)
                                self.assertFalse(any(data.warning.number))
                            self.assertLess(abs(data.qpos[qid]),.08,'active joint did not reopen')

    def test_l6_thumb_root_still_contacts_external_objects(self):
        for side in ['left','right']:
            with self.subTest(side=side):
                path=ASSETS/'linkerhand_l6'/side/'hand.xml'
                spec=mujoco.MjSpec.from_file(str(path))
                # Represent a moving robot-mounted hand and a fixed obstacle.
                spec.body(side[0]+'h_hand_base_link').add_freejoint(name='test_floating_mount')
                model=spec.compile();data=mujoco.MjData(model)
                mujoco.mj_forward(model,data)
                bid=model.body(side[0]+'h_thumb_metacarpals').id
                gid=next(g for g in range(model.ngeom) if model.geom_bodyid[g]==bid and model.geom_contype[g])
                mesh=model.geom_dataid[gid]
                vertices=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
                vertices=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
                # Approach an outer surface; avoid a degenerate probe placed
                # exactly at the convex solver's interior origin.
                pos=vertices[np.argmax(vertices[:,0])]+[.005,0,0]
                spec.worldbody.add_geom(name='external_probe',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.01,0,0],pos=pos)
                model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
                probe=model.geom('external_probe').id
                self.assertTrue(any(probe in (c.geom1,c.geom2) and any(model.geom_bodyid[g]==model.body(side[0]+'h_thumb_metacarpals').id for g in (c.geom1,c.geom2)) for c in data.contact))

    def test_contacts_obstruction_and_release(self):
        jobs=[(h,s) for h in ['linkerhand_o6','linkerhand_l6','linkerhand_l30'] for s in ['left','right']]
        for job in jobs:
            for row in run(job):
                with self.subTest(hand=row['hand'],side=row['side'],mode=row['mode']):
                    self.assertTrue(row['passed'],json.dumps(row))

if __name__=='__main__':
    unittest.main()
