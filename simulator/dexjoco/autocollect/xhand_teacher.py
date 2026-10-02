"""XHand kinematic adapter for the existing privileged geometric teacher."""
import mujoco
import numpy as np
from .linker_teacher import HandFK,FingerRetargeter,XML
from ..sim.envs.xhand_specs import JOINT_NAMES,TIP_BODY_NAMES


class XHandFK(HandFK):
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(XML/'panda_xhand_right.xml'))
        self.data=mujoco.MjData(self.model)
        self.jids=np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.qids=self.model.jnt_qposadr[self.jids]
        self.vids=self.model.jnt_dofadr[self.jids]
        self.low,self.high=self.model.jnt_range[self.jids].T.copy()
        self.bodies=[self.model.body(n).id for n in TIP_BODY_NAMES]
        self.points=np.zeros((5,3))  # source URDF's explicit fingertip marker frames
        self.mimics=[]
        self.site=self.model.site('attachment_site').id


class XHandRetargeter(FingerRetargeter):
    def __init__(self,tcp_offset=(0,0,-.008),squeeze_m=0.):
        super().__init__(tcp_offset,squeeze_m,target_hand=XHandFK(),
                         initial_pose=[.7,.3,.3,0,.3,.3,.3,.3,.3,.3,.3,.3])
