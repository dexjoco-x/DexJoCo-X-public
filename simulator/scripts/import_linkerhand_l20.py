#!/usr/bin/env python3
"""Convert the user-supplied L20 V10.1 URDF package without changing task objects."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from linkerhand_l20_visuals import VERSION as VISUAL_VERSION, REFERENCE as VISUAL_REFERENCE, MATERIAL, add_material, create_visual_obj

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "dexjoco/dexjoco/sim/envs/xmls"
MOUNT_VERSION = 'l20_right_angle_adapter_v1'


def numbers(x):
    return " ".join(f"{v:.12g}" for v in x)


def quat(rpy):
    q = Rotation.from_euler("xyz", np.fromstring(rpy, sep=" ")).as_quat()
    return numbers(q[[3, 0, 1, 2]])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_xml(path, root):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def convert(package, side):
    source = package / f"linkerhand_L20 V10.1_{side}.urdf"
    urdf = source / f"linkerhand_L25NS_{side}.urdf.urdf"
    robot = ET.parse(urdf).getroot()
    target = XML / "linkerhand_l20" / side
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(urdf, target / "source.urdf")
    shutil.copytree(source / "meshes", target / "meshes", dirs_exist_ok=True)
    joints = robot.findall("joint")
    links = {l.attrib["name"]: l for l in robot.findall("link")}
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").attrib["link"], []).append(j)
    root = ET.Element("mujoco", model=f"linkerhand_l20_{side}")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    defaults = ET.SubElement(ET.SubElement(root, "default"), "default", {"class": "linker_l20"})
    ET.SubElement(defaults, "joint", damping="0.05", armature="0.002", limited="true")
    ET.SubElement(defaults, "geom", friction="1 0.005 0.0001", condim="4", solref="0.004 1")
    assets, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    add_material(assets)
    equality, actuators = ET.SubElement(root, "equality"), ET.SubElement(root, "actuator")
    active, limits, mimics, inertia_repairs = [], [], [], []
    visual_components = {}
    prefix = "l20_"

    def add_link(name, parent, joint=None):
        attrs = {"name": prefix + name, "childclass": "linker_l20"}
        if joint is not None:
            origin = joint.find("origin")
            attrs.update(pos=origin.attrib.get("xyz", "0 0 0"), quat=quat(origin.attrib.get("rpy", "0 0 0")))
        body = ET.SubElement(parent, "body", attrs)
        link = links[name]
        inertial = link.find("inertial")
        i = inertial.find("inertia").attrib
        tensor = np.array([[float(i['ixx']),float(i['ixy']),float(i['ixz'])],
            [float(i['ixy']),float(i['iyy']),float(i['iyz'])],
            [float(i['ixz']),float(i['iyz']),float(i['izz'])]])
        origin = inertial.find("origin").attrib
        rotation = Rotation.from_euler("xyz",np.fromstring(origin.get("rpy","0 0 0"),sep=" ")).as_matrix()
        tensor = rotation @ tensor @ rotation.T
        if np.min(np.linalg.eigvalsh(tensor)) <= 0:
            raise ValueError(f"invalid inertia: {name}")
        values, axes = np.linalg.eigh(tensor)
        violation = values[2] - values[0] - values[1]
        if violation > 0:
            # Nearest triangle-valid eigenvalues in Euclidean/Frobenius norm.
            corrected = values + (violation / 3 + 1e-14) * np.array([1, 1, -1])
            inertia_repairs.append(dict(link=name, original_principal=values.tolist(),
                                        corrected_principal=corrected.tolist()))
            tensor = (axes * corrected) @ axes.T
        ET.SubElement(body,"inertial",pos=origin['xyz'],mass=inertial.find("mass").attrib['value'],
            fullinertia=numbers([tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]]))
        for kind in ("visual", "collision"):
            shape = link.find(kind)
            mesh = shape.find("geometry/mesh")
            mesh_name = prefix + name
            if kind == "visual":
                ET.SubElement(assets,"mesh",name=mesh_name,file=f"linkerhand_l20/{side}/meshes/{Path(mesh.attrib['filename']).name}",scale=mesh.attrib.get("scale","1 1 1"))
                visual_components[name] = create_visual_obj(
                    target/'meshes'/Path(mesh.attrib['filename']).name,
                    target/'visual'/f'{name}.obj', name)
                ET.SubElement(assets, 'mesh', name=mesh_name+'_surface',
                    file=f'linkerhand_l20/{side}/visual/{name}.obj', scale=mesh.attrib.get('scale','1 1 1'))
            o = shape.find("origin").attrib
            color = shape.find("material/color")
            geom = ET.SubElement(body,"geom",name=prefix+name+"_"+kind,type="mesh",mesh=mesh_name,
                pos=o.get('xyz','0 0 0'),quat=quat(o.get('rpy','0 0 0')),
                contype="0" if kind=="visual" else "1",conaffinity="0" if kind=="visual" else "1",
                group="1" if kind=="visual" else "3",density="0",
                rgba=color.attrib['rgba'] if color is not None else "0.6 0.6 0.6 1")
            if kind == 'visual':
                geom.set('mesh', mesh_name+'_surface')
                geom.set('material', MATERIAL)
                geom.set('rgba', '1 1 1 1')
        if joint is not None:
            if joint.attrib['type'] != 'revolute':
                raise ValueError('unsupported joint type')
            jname = prefix + joint.attrib['name']
            limit = joint.find('limit').attrib
            bounds = [float(limit['lower']),float(limit['upper'])]
            ET.SubElement(body,'joint',name=jname,type='hinge',axis=joint.find('axis').attrib['xyz'],range=numbers(bounds))
            mimic = joint.find('mimic')
            if mimic is not None:
                multiplier=float(mimic.attrib.get('multiplier',1)); offset=float(mimic.attrib.get('offset',0))
                parent_name=prefix+mimic.attrib['joint']
                ET.SubElement(equality,'joint',joint1=jname,joint2=parent_name,
                    polycoef=numbers([offset,multiplier,0,0,0]),solref='0.004 1')
                mimics.append(dict(joint=jname,parent=parent_name,multiplier=multiplier,offset=offset))
            else:
                effort=float(limit['effort'])
                ET.SubElement(actuators,'position',name=jname+'_ctrl',joint=jname,kp='20',kv='0.2',
                    ctrlrange=numbers(bounds),forcerange=numbers([-effort,effort]))
                active.append(jname); limits.append(bounds)
        for child in children.get(name,[]):
            add_link(child.find('child').attrib['link'],body,child)
        return body

    add_link('hand_base_link',world)
    contact = ET.SubElement(root, 'contact')
    # The CAD palm is hollow around the embedded motor housings. Its convex
    # collision hull fills these slots, creating false contacts at zero pose.
    # Exclude only the embedded housings/knuckles, not finger pads or objects.
    embedded = ['thumb_metacarpals_base1', 'thumb_metacarpals_base2'] + [
        f'{finger}_{part}' for finger in ('index', 'middle', 'ring', 'pinky')
        for part in ('metacarpals', 'proximal')]
    for name in embedded:
        ET.SubElement(contact, 'exclude', body1=prefix+'hand_base_link', body2=prefix+name)
    write_xml(XML/f'linkerhand_l20_{side}.xml',root)
    metadata=dict(package_label='L20 V10.1',urdf_robot_name=robot.attrib['name'],side=side,
        source_sha256={str(p.relative_to(target)):sha(p) for p in target.rglob('*')
                       if p.is_file() and (p == target/'source.urdf' or target/'meshes' in p.parents)},
        joint_names=active,joint_limits=limits,mimic_joints=mimics,physical_joint_count=len(joints),
        independent_action_dim=len(active),position_kp=20,position_kv=.2,
        inertia_repairs=inertia_repairs, palm_housing_collision_exclusions=embedded,
        visual_appearance=dict(version=VISUAL_VERSION,reference=VISUAL_REFERENCE,
            components=visual_components, collision_meshes_unchanged=True,
            note='Approximate silver/graphite appearance from official product images, not a measured reflectance calibration.'),
        note='Joint limits, masses, meshes and mimic ratios come from the supplied URDF. Invalid inertia tensors receive the explicitly recorded minimal principal-moment projection. Servo gains are simulation settings, not manufacturer calibration. Mesh collisions use MuJoCo convex hulls.')
    (target/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return root,metadata


def add_wrist_adapter(model, attachment):
    """Approximate fixed bracket, not a manufacturer CAD or inertia calibration.

    Like the original panda_allegro_copy.xml, the hand stays side-mounted by
    Ry(-pi/2). Coordinates are in the Panda attachment frame (metres). The
    flange face is z=0; the L20 rear mounting plane is x=-0.009, centred at
    z=0.026. The three overlapping primitives form one rigid connector.
    """
    asset = model.find('asset')
    ET.SubElement(asset, 'material', name='l20_adapter_anodized',
                  rgba='0.18 0.20 0.23 1', specular='0.35', shininess='0.35')
    ET.SubElement(asset, 'material', name='l20_adapter_metal',
                  rgba='0.42 0.44 0.47 1', specular='0.45', shininess='0.4')
    adapter = ET.SubElement(attachment, 'body', name='l20_wrist_adapter')
    adapter.append(ET.Comment(
        f' {MOUNT_VERSION}: fixed simulation approximation; 120 g estimated '
        'mass and box inertia, not measured hardware. Hand/TCP transform unchanged. '))
    # Independent explicit inertia avoids double-counting intersecting solids.
    # Approximate the assembly by a 59 x 59 x 51 mm, 120 g bounding box.
    mass = .12
    dimensions = np.array([.059, .059, .051])
    inertia = mass / 12 * (np.sum(dimensions**2) - dimensions**2)
    ET.SubElement(adapter, 'inertial', mass=str(mass), pos='0.004 0 0.024',
                  diaginertia=numbers(inertia))
    solids = [
        ('flange', 'cylinder', '0 0 0.003', '0.0295 0.003', 'l20_adapter_metal'),
        ('elbow', 'box', '0.012 0 0.026', '0.015 0.021 0.020', 'l20_adapter_anodized'),
        ('hand_plate', 'box', '-0.006 0 0.026', '0.003 0.027 0.025', 'l20_adapter_metal'),
    ]
    for name, kind, pos, size, material in solids:
        for purpose in ('visual', 'collision'):
            visual = purpose == 'visual'
            geom = ET.SubElement(adapter, 'geom', name=f'l20_adapter_{name}_{purpose}',
                type=kind, pos=pos, size=size, density='0',
                contype='0' if visual else '1', conaffinity='0' if visual else '1',
                group='1' if visual else '3', friction='0.8 0.005 0.0001',
                condim='4', solref='0.004 1')
            if visual:
                geom.set('material', material)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    args=parser.parse_args()
    converted={side:convert(args.package,side) for side in ('right','left')}
    hand,meta=converted['right']
    model=ET.parse(XML/'panda_inspire_right.xml').getroot()
    model.set('model','panda_linkerhand_l20_right')
    attachment=model.find(".//body[@name='attachment']")
    for child in list(attachment):
        if child.tag in ('frame','body','geom'):
            attachment.remove(child)
    add_wrist_adapter(model, attachment)
    mounted=copy.deepcopy(hand.find('worldbody/body'))
    mounted.set('pos','-0.009 0 0.026')
    mounted.set('quat','0.7071067811865476 0 -0.7071067811865476 0')
    attachment.append(mounted)
    for asset in list(model.find('asset')):
        if asset.attrib.get('name','').startswith('inspire_'):
            model.find('asset').remove(asset)
    model.find('asset').extend(copy.deepcopy(list(hand.find('asset'))))
    model.find('default').extend(copy.deepcopy(list(hand.find('default'))))
    for section in ('equality','actuator'):
        parent=model.find(section)
        for child in list(parent):
            if any(str(v).startswith('inspire_') for v in child.attrib.values()):
                parent.remove(child)
        parent.extend(copy.deepcopy(list(hand.find(section))))
    contact = model.find('contact')
    if contact is None:
        contact = ET.SubElement(model, 'contact')
    contact.extend(copy.deepcopy(list(hand.find('contact'))))
    for key in model.findall('keyframe'):
        model.remove(key)
    write_xml(XML/'panda_linkerhand_l20_right.xml',model)
    for task in ('hammer_nail','bucket_pick'):
        arena=ET.parse(XML/f'arena_arm_hand_{task}_inspire.xml').getroot()
        arena.set('model',f'Arena_Linker_L20_{task}')
        for include in arena.findall('include'):
            if include.attrib['file']=='panda_inspire_right.xml':
                include.set('file','panda_linkerhand_l20_right.xml')
        write_xml(XML/f'arena_arm_hand_{task}_linkerhand_l20.xml',arena)
    print(json.dumps({k:{'joints':v[1]['physical_joint_count'],'actions':v[1]['independent_action_dim']} for k,v in converted.items()}))


if __name__=='__main__':
    main()
