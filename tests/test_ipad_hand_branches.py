"""Grasp branches must stay distinct below a shared actuated wrist."""
from pathlib import Path
from types import SimpleNamespace
import sys, unittest
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simulator'))
from dexjoco.benchmark6.ipad_task import Measurements

class NativePalmBranchTests(unittest.TestCase):
    def test_fingers_are_distinct_below_palm(self):
        for hand in ('linkerhand_o6','linkerhand_l6','linkerhand_l30'):
            with self.subTest(hand=hand):
                path=ROOT/'simulator/dexjoco/sim/envs/xmls'/hand/'left/hand.xml'
                donor=mujoco.MjSpec.from_file(str(path))
                for mesh in donor.meshes:
                    if mesh.file:mesh.file=str((path.parent/donor.compiler.meshdir/mesh.file).resolve())
                donor.compiler.meshdir=''
                spec=mujoco.MjSpec();spec.copy_during_attach=True;spec.compiler.degree=False
                spec.worldbody.add_frame().attach_body(donor.body('lh_hand_base_link'),prefix='benchhand_left_native_',suffix='')
                ipad=spec.worldbody.add_body(name='ipad_body',pos=[1,0,0])
                ipad.add_geom(name='ipad_screen',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.1,.1,.01])
                model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
                palm='benchhand_left_native_lh_'+('wrist_carpal_link0' if hand=='linkerhand_l30' else 'hand_base_link')
                env=SimpleNamespace(_model=model,_data=data,hand_metadata={'sides':{'left':{'palm_body_name':palm}}})
                meter=Measurements(env);branches=[]
                for finger in ('thumb','index','middle','ring','pinky'):
                    body=model.body('benchhand_left_native_lh_'+finger+'_distal').id
                    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]==body]
                    ids={meter.branches[g] for g in geoms}
                    self.assertEqual(len(ids),1);branches.append(ids.pop())
                self.assertEqual(len(set(branches)),5)
                palm_geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]==model.body(palm).id]
                self.assertTrue(palm_geoms)
                self.assertTrue(all(meter.branches[g]==-1 for g in palm_geoms))
                if hand=='linkerhand_l30':
                    base=model.body('benchhand_left_native_lh_hand_base_link').id
                    self.assertTrue(all(g not in meter.branches for g in range(model.ngeom) if model.geom_bodyid[g]==base))
                    # The legacy path remains available for existing models.
                    legacy=Measurements(SimpleNamespace(_model=model,_data=data))
                    old={legacy.branches[g] for g in legacy.branches if model.body(model.geom_bodyid[g]).name.endswith('_distal')}
                    self.assertEqual(len(old),1)

if __name__=='__main__':unittest.main()
