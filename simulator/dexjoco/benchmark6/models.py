"""Compose native hand subtrees onto the SAME task-specific Franka reference.

Names beginning ``benchhand_`` are the only permitted morphology-dependent
model elements. Wrist cameras/TCPs, arms and task objects remain in the parent.
MjSpec attachment retains native tendons, mimic constraints and actuators.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import VERSION, HANDS, CANDIDATE_TASKS

XML = Path(__file__).resolve().parents[1] / "sim/envs/xmls"
PREFIX = "benchhand_"
DONORS = {'allegro': (('panda_allegro_copy.xml', 'attachment'), ('panda_allegro_left.xml', 'attachment_left')), 'leap': (('panda_leap_right.xml', 'attachment'), ('panda_leap_left.xml', 'attachment')), 'inspire': (('panda_inspire_right.xml', 'attachment'), ('panda_inspire_bimanual_left.xml', 'attachment_left')), 'wuji': (('panda_wuji_right.xml', 'attachment'), ('panda_wuji_left.xml', 'attachment')), 'sharpa_wave': (('panda_sharpa_right.xml', 'attachment'), ('panda_sharpa_left.xml', 'attachment')), 'xhand': (('panda_xhand_right.xml', 'attachment'), ('xhand_left.xml', None)), 'linkerhand_l20': (('panda_linkerhand_l20_right.xml', 'attachment'), ('linkerhand_l20_left.xml', None))}


def path_for(task, hand):
    if task == 'single_hand_hanoi' and hand in HANDS:
        return XML / f'{VERSION}_single_hand_hanoi_inspire_scene_{hand}.xml'
    if task not in CANDIDATE_TASKS or hand not in HANDS:
        raise ValueError((task, hand))
    return XML / f"{VERSION}_{task}_{hand}.xml"


def _absolute_files(spec):
    # Donors and generated XMLs live in XML, including expanded include paths.
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str((XML / spec.compiler.meshdir / mesh.file).resolve())
    for texture in spec.textures:
        if texture.file:
            texture.file = str((XML / spec.compiler.texturedir / texture.file).resolve())
    spec.compiler.meshdir = ""
    spec.compiler.texturedir = ""


def extract_hand(hand, side):
    filename, attachment = DONORS[hand][side == "left"]
    source = mujoco.MjSpec.from_file(str(XML / filename))
    _absolute_files(source)
    out = mujoco.MjSpec()
    out.copy_during_attach = True
    out.compiler.degree = False
    if attachment is not None:
        frame = out.worldbody.add_frame()
        root = frame.attach_body(source.body(attachment), prefix="native_", suffix="")
        root.name = "mount"
        root.pos = [0, 0, 0]
        root.quat = [1, 0, 0, 0]
        for sensor in list(out.sensors):
            out.delete(sensor)
        for item in list(root.sites) + list(root.cameras):
            out.delete(item)
    elif hand == "linkerhand_l20":
        # Genuine left CAD, not a mirrored mesh. Its native base frame uses
        # the same +Z finger direction and rear mounting plane as the right.
        root = out.worldbody.add_body(name="mount")
        right = extract_hand(hand, "right")
        root.add_frame().attach_body(right.body("native_l20_wrist_adapter"),
                                     prefix="adapter_", suffix="")
        cad = root.add_frame().attach_body(source.body("l20_hand_base_link"),
                                           prefix="native_", suffix="")
        cad.pos = [-.009, 0, .026]
        cad.quat = [np.sqrt(.5), 0, -np.sqrt(.5), 0]
    else:
        # Use the genuine LEFT CAD, not a reflected right hand. Cancel its
        # URDF fixed base convention, using the same physical mount as right.
        root = out.worldbody.add_body(name="mount")
        right = extract_hand("xhand", "right")
        frame = root.add_frame()
        frame.attach_body(right.body("native_xhand_wrist_adapter"), prefix="adapter_", suffix="")
        urdf = ET.parse(XML / "xhand/left/source.urdf").getroot()
        fixed = next(j for j in urdf.findall("joint")
                     if j.find("parent").get("link") == "base_link")
        origin = fixed.find("origin")
        transform = np.eye(4)
        transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        desired = np.eye(4)
        desired[:3, 3] = [-.025, 0, .030]
        # The genuine left CAD needs an additional rx(pi) relative to the
        # right-hand convention.  Without it the complete left hand is
        # mounted 180 degrees away even though joint and actuator interfaces
        # still look valid.  This is the validated xhand_left_mount_rx180_v1
        # orientation used by the released XHand iPad demonstrations.
        desired[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        pose = desired @ np.linalg.inv(transform)
        cad = root.add_frame().attach_body(source.body("xhand_left_base_link"), prefix="native_", suffix="")
        cad.pos = pose[:3, 3]
        cad.quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    from .asset_repairs import repair
    repair(out, hand, side)
    out.compile()
    return out


def _hand_metadata(model, side):
    prefix = PREFIX + side + "_"
    jids = [i for i in range(model.njnt) if model.joint(i).name.startswith(prefix)]
    aids = [i for i in range(model.nu) if model.actuator(i).name.startswith(prefix)]
    # Mimic joints are observable, but are not independent command channels.
    dependent = {int(model.eq_obj1id[e]) for e in range(model.neq)
                 if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT}
    kinematic = [i for i in jids if i not in dependent]
    commands = []
    transmissions = []
    for aid in aids:
        kind = model.actuator_trntype[aid]
        if kind == mujoco.mjtTrn.mjTRN_JOINT:
            joint = int(model.actuator_trnid[aid, 0])
            terms = [(joint, float(model.actuator_gear[aid, 0]))]
        elif kind == mujoco.mjtTrn.mjTRN_TENDON:
            tid = int(model.actuator_trnid[aid, 0]); start = int(model.tendon_adr[tid])
            terms = []
            for wrap in range(start, start + int(model.tendon_num[tid])):
                if model.wrap_type[wrap] != mujoco.mjtWrap.mjWRAP_JOINT:
                    raise ValueError("Only fixed joint tendons supported")
                terms.append((int(model.wrap_objid[wrap]), float(model.wrap_prm[wrap])))
            joint = terms[0][0]
        else:
            raise ValueError(f"Unsupported transmission {kind}")
        commands.append(joint)
        transmissions.append([(model.joint(j).name, coefficient) for j, coefficient in terms])
    if len(set(commands)) != len(commands):
        raise ValueError("Reset proxy joints must be unique")
    home = np.clip(np.zeros(len(commands)), model.jnt_range[commands, 0], model.jnt_range[commands, 1])
    names = [model.joint(i).name for i in commands]
    for i, name in enumerate(names):
        if "wuji_F" in name:
            for finger, spread in ((2, .18), (3, 0), (4, .14), (5, .20)):
                if f"F{finger}J2" in name:
                    home[i] = np.clip(spread, *model.jnt_range[commands[i]])
    return dict(joint_names=names, state_joint_names=[model.joint(i).name for i in jids],
                kinematic_joint_names=[model.joint(i).name for i in kinematic],
                transmissions=transmissions,
                actuator_names=[model.actuator(i).name for i in aids], home=home.tolist(),
                joint_ranges=model.jnt_range[commands].tolist(),
                actuator_ranges=model.actuator_ctrlrange[aids].tolist(),
                independent_commands=len(aids), actuators=len(aids), state_joints=len(jids),
                command_semantics="actuator joint/tendon position targets; proxy joint_names used only for reset")


def build(task, hand):
    if task == 'single_hand_hanoi':
        import json
        return json.loads(path_for(task, hand).with_suffix('.json').read_text())
    if VERSION == 'benchmark7_v3':
        from .geometry_revision import build as build_revision
        return build_revision(task, hand)
    if VERSION == 'benchmark7_v2':
        # Revise the preserved v1 XML directly: do not regenerate hand assets
        # or disturb existing geometric/visual/native-dynamics repairs.
        from .contact_revision import build as build_revision
        return build_revision(task, hand)
    replacements = {"click_mouse": "arena_arm_hand_monitor_mouse.xml",
                    "water_plant": "arena_arm_hand_plant_inspire.xml",
                    "bimanual_hanoi": "arena_arm_hand_bimanual_hanoi.xml"}
    reference = XML / replacements.get(task, f"cross_embodiment_v1_{task}_inspire.xml")
    spec = mujoco.MjSpec.from_file(str(reference))
    spec.copy_during_attach = True
    _absolute_files(spec)
    for key in list(spec.keys):
        spec.delete(key)
    sides = ("right", "left") if task.startswith("bimanual_") else ("right",)
    for side in sides:
        rootname = "attachment" if len(sides) == 1 else "attachment_" + side
        root = spec.body(rootname)
        # Remove just the old hand. Keep the reference TCP, camera, wrist pose.
        for item in list(root.bodies) + list(root.geoms):
            spec.delete(item)
        # Empty coordinate frames are harmless; deleting both their contents
        # and the frame triggers a MuJoCo 3.4 MjSpec lifetime bug on teardown.
        child = extract_hand(hand, side)
        root.add_frame().attach_body(child.body("mount"), prefix=PREFIX + side + "_", suffix="")
        spec.compile()  # resolve references before attaching another subtree
    spec.modelname = f"{VERSION}_{task}_{hand}"
    model = spec.compile()
    path = path_for(task, hand)
    serialized = spec.to_xml()
    path.write_text(serialized)
    # Reopen serialized artifact: do not only validate the in-memory spec.
    reopened = mujoco.MjModel.from_xml_path(str(path))
    if (model.nq, model.nv, model.nu) != (reopened.nq, reopened.nv, reopened.nu):
        raise RuntimeError("Serialized model dimensions differ")
    meta = dict(version=VERSION, task=task, hand=hand, status="candidate_not_task_validated",
                reference=str(reference), reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sides={side: _hand_metadata(reopened, side) for side in sides})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
