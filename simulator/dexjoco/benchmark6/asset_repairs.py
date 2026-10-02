"""Documented hand-only repairs; source assets and legacy task models stay intact."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parents[1] / "sim/envs/xmls/benchmark6_assets"


@lru_cache(None)
def inspire_visual_parts(link, side):
    """Recover the original GLB's per-component materials lost in flattened OBJ.

    Verify the transformed triangles match the old visual geometry before using
    the color restoration. LEFT legacy model mirrors Y; use exactly that mirror.
    Collision meshes are NEVER changed by this function.
    """
    base = ROOT / "teleoperation/inspire/assets/right"
    source = base / "meshes/visual" / f"right_{link}.glb"
    obj = base / f"meshes__visual__right_{link}.obj"
    old = trimesh.load(obj, force="mesh", process=False)
    scene = trimesh.load(source, force="scene")
    parts = []
    centers = []
    for node in sorted(scene.graph.nodes_geometry):
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy(); mesh.apply_transform(transform)
        centers.append(mesh.triangles_center.copy())
        mat = getattr(mesh.visual, "material", None)
        rgba = getattr(mat, "baseColorFactor", None)
        rgba = np.ones(4) if rgba is None else np.asarray(rgba)/255.
        if getattr(mat, "baseColorTexture", None) is not None:
            raise ValueError("Unexpected GLB texture: cannot drop it")
        if side == "left": mesh.apply_transform(np.diag([1., -1., 1., 1.]))
        parts.append((mesh, rgba))
    centers = np.concatenate(centers)
    if len(centers) != len(old.faces): raise ValueError(f"Triangle count changed: {link}")
    error = max(cKDTree(centers).query(old.triangles_center)[0].max(),
                cKDTree(old.triangles_center).query(centers)[0].max())
    if error > 2e-7: raise ValueError(f"GLB/OBJ geometry mismatch {link}: {error}")
    directory = ASSETS / "inspire" / side; directory.mkdir(parents=True, exist_ok=True)
    result = []
    for index, (mesh, rgba) in enumerate(parts):
        path = directory / f"{link}_{index}.obj"
        # Solid material is assigned by MJCF; do not export original MTL paths.
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        mesh.export(path, include_color=False, include_texture=False)
        result.append(dict(path=str(path), rgba=rgba.tolist(), faces=len(mesh.faces)))
    (directory/f"{link}.json").write_text(json.dumps(dict(
        glb=str(source), glb_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_obj=str(obj), original_obj_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(),
        geometry_max_centroid_error_m=float(error), parts=result), indent=2)+"\n")
    return result


def restore_inspire_colors(spec, side):
    for geom_number, geom in enumerate(list(spec.geoms)):
        if not geom.meshname or "meshes__visual__" not in geom.meshname or geom.contype or geom.conaffinity:
            continue
        if "base_link_collision" in geom.name: continue
        old_mesh = spec.mesh(geom.meshname)
        # Filename is independent of prefixed model namespaces and left suffix.
        filename = Path(old_mesh.file).stem
        marker = "left_mirrored_" if side == "left" else "right_"
        link = filename.split(marker, 1)[1]
        parts = inspire_visual_parts(link, side)
        position, quat = geom.pos.copy(), geom.quat.copy()
        parent, frame = geom.parent, geom.frame
        for index, part in enumerate(parts):
            name = f"restored_{side}_{link}_{index}"
            if spec.mesh(name) is None:
                mesh_asset = spec.add_mesh(name=name, file=part["path"])
                mesh_asset.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                material = spec.add_material(name=name+"_material")
                material.rgba = part["rgba"]; material.specular = .15; material.shininess = .2
            target = geom if index == 0 else parent.add_geom()
            if frame is not None: target.set_frame(frame)
            target.name = name+f"_visual_{geom_number}"
            target.type = mujoco.mjtGeom.mjGEOM_MESH
            target.meshname = name; target.material = name+"_material"
            target.pos = position; target.quat = quat
            target.contype = 0; target.conaffinity = 0; target.group = 1
            target.density = 0; target.rgba = part["rgba"]


def restore_palm_inertia(spec, hand, side):
    """Restore palm inertia dropped during old frame-based mounting.

    The native URDF mass properties remain an approximation (not calibrated
    hardware masses). An explicitly labeled 120g adapter is added analytically.
    """
    if hand not in ("inspire", "wuji"): return
    urdf_side = "right" if hand == "inspire" else side
    urdf = ROOT / f"teleoperation/{hand}/assets/{urdf_side}/main_mujoco.urdf"
    tree = ET.parse(urdf).getroot()
    name = "hand_base_link" if hand == "inspire" else "base_link"
    inertial = next(l for l in tree.findall("link") if l.get("name") == name).find("inertial")
    mass = float(inertial.find("mass").get("value"))
    com = np.fromstring(inertial.find("origin").get("xyz"), sep=" ")
    a = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
    tensor = np.array([[a["ixx"], a["ixy"], a["ixz"]],
                       [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]])
    rpy = np.fromstring(inertial.find("origin").get("rpy", "0 0 0"), sep=" ")
    rot = Rotation.from_euler("xyz", rpy).as_matrix(); tensor = rot @ tensor @ rot.T
    if hand == "inspire" and side == "left":
        mirror = np.diag([1, -1, 1]); com = mirror @ com; tensor = mirror @ tensor @ mirror
    root = spec.body("mount")
    palm = next(g for g in root.geoms if "base_link" in g.meshname)
    rot = Rotation.from_quat(palm.quat[[1,2,3,0]]).as_matrix()
    com = palm.pos + rot @ com; tensor = rot @ tensor @ rot.T
    adapter = next(g for g in root.geoms if "wrist_adapter_visual" in g.name)
    adapter.material = ""; adapter.rgba = [.20, .21, .23, 1]
    adapter.density = 0
    amass = .12; acomp = adapter.pos.copy(); half = adapter.size.copy()
    atensor = np.diag(amass/3 * np.array([half[1]**2+half[2]**2, half[0]**2+half[2]**2, half[0]**2+half[1]**2]))
    total = mass + amass; center = (mass*com + amass*acomp)/total
    inertia = np.zeros((3,3))
    for weight, point, intrinsic in ((mass, com, tensor), (amass, acomp, atensor)):
        delta = point-center
        inertia += intrinsic + weight*(np.eye(3)*np.dot(delta, delta)-np.outer(delta,delta))
    root.mass = total; root.ipos = center; root.iquat = [1,0,0,0]
    root.fullinertia = inertia[[0,1,2,0,0,1], [0,1,2,1,2,2]]
    root.explicitinertial = True
    collision = root.add_geom(name=f"{hand}_wrist_adapter_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                              pos=adapter.pos, quat=adapter.quat, size=adapter.size,
                              contype=1, conaffinity=1, group=3, density=0)
    collision.rgba = [.20,.21,.23,1]
    # This root explicit inertia intentionally replaces auto-inferred geom mass.
    return dict(source=str(urdf), palm_mass=mass, adapter_mass=amass,
                approximation="source URDF inertia + 120g analytic box adapter, not hardware-calibrated")


def restore_leap_adapter_collision(spec, hand="leap"):
    """Give the existing visible bracket matching collision, without extra mass.

    Keep the donor's 134.4 g auto-inferred box inertia. A duplicate density=0
    collision geom must not double it. Palm, fingers and mounting poses stay
    unchanged; this is only enabled in the new benchmark model revision.
    """
    marker = "sharpa" if hand == "sharpa_wave" else "leap"
    visual = next(g for g in spec.geoms if marker + "_wrist_adapter_visual" in g.name)
    parent = visual.parent
    collision = parent.add_geom(
        name=marker + "_wrist_adapter_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=visual.pos, quat=visual.quat, size=visual.size,
        contype=1, conaffinity=1, group=3, density=0)
    if visual.frame is not None:
        collision.set_frame(visual.frame)
    collision.rgba = visual.rgba


def repair(spec, hand, side):
    inertia = restore_palm_inertia(spec, hand, side)
    if hand in ("leap", "sharpa_wave"): restore_leap_adapter_collision(spec, hand)
    if hand == "inspire": restore_inspire_colors(spec, side)
    # MJCF's ctrlrange=[0,0] means UNDEFINED if ctrllimited=false, not a
    # physically locked actuator. Make the position-control bounds explicit
    # from native joint/tendon ranges before exporting a bounded action space.
    model = spec.compile()
    for aid in range(model.nu):
        if model.actuator_ctrllimited[aid]: continue
        kind = model.actuator_trntype[aid]
        if kind == mujoco.mjtTrn.mjTRN_JOINT:
            terms = [(int(model.actuator_trnid[aid,0]), float(model.actuator_gear[aid,0]))]
        elif kind == mujoco.mjtTrn.mjTRN_TENDON:
            tid = int(model.actuator_trnid[aid,0]); start = int(model.tendon_adr[tid])
            terms = []
            for wrap in range(start,start+int(model.tendon_num[tid])):
                if model.wrap_type[wrap] != mujoco.mjtWrap.mjWRAP_JOINT:
                    raise ValueError("Only joint tendons can have inferred position bounds")
                terms.append((int(model.wrap_objid[wrap]),float(model.wrap_prm[wrap])))
        else: raise ValueError("Unsupported actuator transmission")
        limits = np.zeros(2)
        for joint, coefficient in terms:
            if not model.jnt_limited[joint]: raise ValueError("Cannot infer finite actuator bounds")
            limits += np.sort(model.jnt_range[joint]*coefficient)
        actuator = spec.actuator(model.actuator(aid).name)
        actuator.ctrlrange = limits; actuator.ctrllimited = True
    return inertia
