"""A rigid simulation adapter joining the unchanged flange and wrist frames."""
import numpy as np,mujoco

def add_adapter(root,hand):
    # The hand's source wrist plane is x=-9 mm in the flange frame, and
    # its long axis points along -X. This bracket joins it to the Z flange.
    # Both visible surfaces and contact geometries are present.
    # Keep source inertias unchanged; the rigid adapter geometry adds no
    # unmeasured mass to the manufacturer-provided hand/flange dynamics.
    radius=.025 if hand!='linkerhand_l30' else .030
    parts=[('flange',mujoco.mjtGeom.mjGEOM_CYLINDER,[.027,.004,0],[0,0,.002],[1,0,0,0]),
           ('bridge',mujoco.mjtGeom.mjGEOM_BOX,[.010,.021,.013],[-.002,0,.017],[1,0,0,0]),
           ('wrist_plate',mujoco.mjtGeom.mjGEOM_CYLINDER,[radius,.004,0],[-.007,0,.026],[np.sqrt(.5),0,np.sqrt(.5),0])]
    for name,kind,size,pos,quat in parts:
        for collision in (False,True):
            root.add_geom(name='adapter_'+name+('_collision' if collision else '_visual'),type=kind,size=size,pos=pos,quat=quat,
                          group=3 if collision else 1,contype=int(collision),conaffinity=int(collision),
                          mass=0,friction=[1,.005,.0001],
                          rgba=[.24,.27,.30,1],margin=0)
    for y in (-.016,.016):
        root.add_geom(name='adapter_bolt_'+str(int(y*1000)),type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[.003,.0012,0],pos=[-.002,y,.031],group=1,contype=0,conaffinity=0,mass=0,rgba=[.65,.68,.70,1])
