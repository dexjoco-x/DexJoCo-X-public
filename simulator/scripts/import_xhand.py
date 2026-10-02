#!/usr/bin/env python3
"""Import the local DexLatent XHand URDF assets into DexJoCo (no VLA used)."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
XML=ROOT/'dexjoco/dexjoco/sim/envs/xmls'
VERSION='dexlatent_xhand_v1'
MOUNT_POS=np.array([-.025,0,.030])
# Raw CAD axes: +x thumb side, +y palm face, -z fingers.
# Panda attachment axes: +y thumb side, +z palm face, -x fingers.
PALM_ROT=np.array([[0.,0,1],[1,0,0],[0,1,0]])


def nums(values):return ' '.join(f'{v:.15g}' for v in values)


def transform(origin):
    out=np.eye(4)
    if origin is not None:
        out[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        out[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return out


def pose_attrs(matrix):
    return dict(pos=nums(matrix[:3,3]),quat=nums(Rotation.from_matrix(matrix[:3,:3]).as_quat()[[3,0,1,2]]))


def write_xml(path,root):
    ET.indent(root,space='  ')
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(source,side):
    urdf=source/f'xhand_{side}.urdf';robot=ET.parse(urdf).getroot()
    links={l.get('name'):l for l in robot.findall('link')}
    children={};incoming={}
    for joint in robot.findall('joint'):
        child=joint.find('child').get('link');parent=joint.find('parent').get('link')
        if joint.get('type') not in ('fixed','revolute') or joint.find('mimic') is not None:
            raise ValueError('This converter expects the independent 12-DOF XHand URDF.')
        incoming[child]=joint;children.setdefault(parent,[]).append(child)
    roots=set(links)-set(incoming)
    if roots!={'base_link'}:raise ValueError(f'Unexpected roots: {roots}')
    def name(link):return f'xhand_{side}_base_link' if link=='base_link' else 'xhand_'+link
    clusters={};members={};fixed_transforms={};invalid_fixed=[]
    def group(link,cluster,relative):
        clusters[link]=cluster;fixed_transforms[link]=relative
        inertial=links[link].find('inertial')
        if inertial is not None:
            a={k:float(v) for k,v in inertial.find('inertia').attrib.items()}
            tensor=np.array([[a['ixx'],a['ixy'],a['ixz']],[a['ixy'],a['iyy'],a['iyz']],[a['ixz'],a['iyz'],a['izz']]])
            eig=np.linalg.eigvalsh(tensor)
            if eig[0]<=0 or eig[2]>eig[:2].sum()+1e-14:
                if link==cluster:raise ValueError(f'Invalid moving-link inertia: {link}')
                invalid_fixed.append(dict(link=link,principal=eig.tolist(),merged_into=cluster))
            frame=relative@transform(inertial.find('origin'))
            mass=float(inertial.find('mass').get('value'))
            members.setdefault(cluster,[]).append((link,mass,frame[:3,3],frame[:3,:3]@tensor@frame[:3,:3].T))
        for child in children.get(link,[]):
            j=incoming[child]
            if j.get('type')=='fixed':group(child,cluster,relative@transform(j.find('origin')))
            else:group(child,child,np.eye(4))
    group('base_link','base_link',np.eye(4))
    inertias={}
    for cluster,rows in members.items():
        mass=sum(row[1] for row in rows)
        center=sum(m*p for _,m,p,_ in rows)/mass
        tensor=sum(i+m*(np.eye(3)*np.dot(p-center,p-center)-np.outer(p-center,p-center)) for _,m,p,i in rows)
        eig=np.linalg.eigvalsh(tensor)
        if eig[0]<=0 or eig[2]>eig[:2].sum()+1e-14:raise ValueError(f'Invalid combined inertia: {cluster}')
        inertias[cluster]=(mass,center,tensor)
    target=XML/'xhand'/side;target.mkdir(parents=True,exist_ok=True)
    (target/'meshes').mkdir(exist_ok=True)
    shutil.copy2(urdf,target/'source.urdf')
    root=ET.Element('mujoco',model=f'xhand_{side}')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    defaults=ET.SubElement(ET.SubElement(root,'default'),'default',{'class':'xhand'})
    ET.SubElement(defaults,'joint',damping='.05',armature='.002',limited='true')
    ET.SubElement(defaults,'geom',density='0',friction='1 .005 .0001',condim='4',solref='.004 1')
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');actuators=ET.SubElement(root,'actuator')
    names=[];limits=[];efforts=[];velocities=[];hashes={'source.urdf':sha(urdf)}
    def add(link,parent):
        joint=incoming.get(link)
        body=ET.SubElement(parent,'body',name=name(link),childclass='xhand',
                          **pose_attrs(transform(None if joint is None else joint.find('origin'))))
        if link in inertias:
            mass,c,t=inertias[link]
            ET.SubElement(body,'inertial',mass=str(mass),pos=nums(c),
                fullinertia=nums([t[0,0],t[1,1],t[2,2],t[0,1],t[0,2],t[1,2]]))
        if joint is not None and joint.get('type')=='revolute':
            jname='xhand_'+joint.get('name');lim=joint.find('limit').attrib
            bounds=[float(lim['lower']),float(lim['upper'])];effort=float(lim['effort'])
            ET.SubElement(body,'joint',name=jname,type='hinge',axis=joint.find('axis').get('xyz'),range=nums(bounds))
            ET.SubElement(actuators,'position',name=jname+'_ctrl',joint=jname,kp='20',kv='.2',
                          ctrlrange=nums(bounds),forcerange=nums([-effort,effort]))
            names.append(jname);limits.append(bounds);efforts.append(effort);velocities.append(float(lim['velocity']))
        for purpose in ('visual','collision'):
            for index,shape in enumerate(links[link].findall(purpose)):
                mesh=shape.find('geometry/mesh')
                if mesh is None:raise ValueError(f'Unsupported geometry: {link}')
                src=(source/mesh.get('filename')).resolve()
                if source.resolve() not in src.parents:raise ValueError(f'Asset outside source: {src}')
                dst=target/'meshes'/src.name
                if not dst.exists() or sha(dst)!=sha(src):shutil.copy2(src,dst)
                hashes[str(dst.relative_to(target))]=sha(src)
                meshname=f'{name(link)}_{purpose}_{index}_mesh'
                ET.SubElement(assets,'mesh',name=meshname,file=f'xhand/{side}/meshes/{src.name}',scale=mesh.get('scale','1 1 1'))
                visual=purpose=='visual';color=shape.find('material/color')
                ET.SubElement(body,'geom',name=f'{name(link)}_{purpose}_{index}',type='mesh',mesh=meshname,
                    **pose_attrs(transform(shape.find('origin'))),contype='0' if visual else '1',
                    conaffinity='0' if visual else '1',group='1' if visual else '3',
                    rgba=color.get('rgba') if visual and color is not None else '.3 .3 .3 1')
        for child in children.get(link,[]):add(child,body)
    add('base_link',world)
    # Explicitly apply the usual adjacent-rigid-body collision filter also to
    # the fixed-base standalone hand (MuJoCo treats world-parent pairs specially).
    # This never excludes nonadjacent finger pairs or hand/object collisions.
    contact=ET.SubElement(root,'contact');exclusions=[]
    collision_links={link for link in links if links[link].find('collision') is not None}
    for child,joint in incoming.items():
        if joint.get('type')!='revolute':continue
        parent_cluster=clusters[joint.find('parent').get('link')]
        for a in sorted(collision_links):
            if clusters[a]!=parent_cluster:continue
            for b in sorted(collision_links):
                if clusters[b]!=child:continue
                ET.SubElement(contact,'exclude',body1=name(a),body2=name(b))
                exclusions.append([name(a),name(b)])
    if len(names)!=12:raise ValueError(f'Expected 12 actuated joints, got {len(names)}')
    write_xml(XML/f'xhand_{side}.xml',root)
    meta=dict(version=VERSION,source='Local third_party/DexLatent/Assets/xhand',side=side,
        source_sha256=hashes,urdf_robot_name=robot.get('name'),joint_names=names,joint_limits=limits,
        effort_limits_nm=efforts,urdf_velocity_limits_rad_s=velocities,
        velocity_limits_enforced=False,position_kp=20,position_kv=.2,
        total_hand_mass_kg=sum(i[0] for i in inertias.values()),independent_action_dim=12,
        fixed_joint_count=18,fixed_link_inertia_aggregation={k:[r[0] for r in rows] for k,rows in members.items()},
        source_invalid_fixed_inertias=invalid_fixed,inertia_tensor_repairs=[],
        adjacent_body_collision_exclusions=exclusions,
        note='Fixed accessory inertias are transformed and summed with their rigid parent, without modifying source tensors. All combined inertias are physical. Original URDF/meshes/colors/limits/masses retained. Convex mesh collisions and simulated position gains are not hardware calibration.')
    (target/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return root,meta,transform(incoming[f'{side}_hand_link'].find('origin'))


def mount(hand,source_base_transform):
    model=ET.parse(XML/'panda_inspire_right.xml').getroot();model.set('model','panda_xhand_right')
    attachment=model.find(".//body[@name='attachment']")
    for child in list(attachment):
        if child.tag in ('frame','body','geom'):attachment.remove(child)
    for asset in list(model.find('asset')):
        if asset.get('name','').startswith('inspire_'):model.find('asset').remove(asset)
    model.find('asset').extend(copy.deepcopy(list(hand.find('asset'))))
    model.find('default').extend(copy.deepcopy(list(hand.find('default'))))
    for section in ('equality','actuator'):
        parent=model.find(section)
        for child in list(parent):
            if any(str(v).startswith('inspire_') for v in child.attrib.values()):parent.remove(child)
        if hand.find(section) is not None:parent.extend(copy.deepcopy(list(hand.find(section))))
    contact=model.find('contact')
    if contact is None:contact=ET.SubElement(model,'contact')
    contact.extend(copy.deepcopy(list(hand.find('contact'))))
    adapter=ET.SubElement(attachment,'body',name='xhand_wrist_adapter')
    adapter.append(ET.Comment(' Approximate right-angle bracket, 120 g box inertia; not manufacturer mounting CAD. '))
    ET.SubElement(adapter,'inertial',mass='.12',pos='0 0 .025',diaginertia='0.00006762 0.00006762 0.00006962')
    solids=[('flange','cylinder','0 0 .003','.0295 .003'),
            ('elbow','box','.004 0 .028','.023 .022 .022'),
            ('hand_plate','box','-.022 0 .030','.003 .030 .025')]
    for name,kind,pos,size in solids:
        for purpose in ('visual','collision'):
            visual=purpose=='visual'
            ET.SubElement(adapter,'geom',name=f'xhand_adapter_{name}_{purpose}',type=kind,pos=pos,size=size,
                density='0',contype='0' if visual else '1',conaffinity='0' if visual else '1',
                group='1' if visual else '3',rgba='.2 .22 .25 1',friction='.8 .005 .0001')
    desired=np.eye(4);desired[:3,3]=MOUNT_POS;desired[:3,:3]=PALM_ROT
    base_transform=desired@np.linalg.inv(source_base_transform)
    body=copy.deepcopy(hand.find('worldbody/body'));body.attrib.update(pose_attrs(base_transform));attachment.append(body)
    for key in model.findall('keyframe'):model.remove(key)
    write_xml(XML/'panda_xhand_right.xml',model)
    (XML/'xhand/mount_right.json').write_text(json.dumps(dict(version='xhand_right_angle_adapter_v1',
        reference='panda_allegro_copy.xml side-mounted fixed hand convention',
        raw_palm_position=MOUNT_POS.tolist(),raw_palm_rotation=PALM_ROT.tolist(),
        root_transform=base_transform.tolist(),estimated_adapter_mass_kg=.12,
        note='Uncalibrated simulation bracket. Twelve hand actuators, no added wrist DOF.'),indent=2)+'\n')
    for task in ('hammer_nail','bucket_pick'):
        arena=ET.parse(XML/f'arena_arm_hand_{task}_inspire.xml').getroot()
        arena.set('model',f'Arena_XHand_{task}')
        for include in arena.findall('include'):
            if include.get('file')=='panda_inspire_right.xml':include.set('file','panda_xhand_right.xml')
        write_xml(XML/f'arena_arm_hand_{task}_xhand.xml',arena)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'third_party/DexLatent/Assets/xhand')
    args=parser.parse_args();hands={side:convert(args.source,side) for side in ('right','left')}
    mount(hands['right'][0],hands['right'][2])
    license_path=args.source.parent.parent/'LICENSE'
    if license_path.is_file():shutil.copy2(license_path,XML/'xhand/SOURCE_LICENSE')
    print(json.dumps({s:{k:v[1][k] for k in ('independent_action_dim','total_hand_mass_kg')} for s,v in hands.items()}))


if __name__=='__main__':main()
