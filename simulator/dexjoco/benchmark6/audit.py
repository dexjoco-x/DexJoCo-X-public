"""Compiled-model, name-based comparisons: asset IDs may differ, physics may not."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import mujoco
from .models import PREFIX


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _values(model, stem, fields, index):
    return {field: np.asarray(getattr(model, stem + "_" + field)[index]).tolist()
            for field in fields.split()}


def nonhand_manifest(model):
    """Independent audit including actual mesh arrays and texture pixel hashes.

    Derived aggregate inertias/contact solver matrices depend on hand mass and
    are deliberately not compared. Own arm inertias and all non-hand dynamics
    ARE compared. Do not compare array IDs whose offsets change with morphology.
    """
    result = {k: {} for k in ("bodies", "joints", "geoms", "actuators", "sites", "cameras", "lights")}
    def bname(i): return model.body(int(i)).name or "world"
    def handbody(i): return bname(i).startswith(PREFIX)
    def material(mid):
        if mid < 0: return None
        out = _values(model, "mat", "rgba specular shininess reflectance texrepeat texuniform emission", mid)
        textures = []
        for tid in np.atleast_1d(model.mat_texid[mid]):
            if tid < 0: textures.append(None); continue
            tid = int(tid); start = int(model.tex_adr[tid])
            length = int(model.tex_height[tid] * model.tex_width[tid] * model.tex_nchannel[tid])
            tex = _values(model, "tex", "type height width nchannel", tid)
            tex["pixels_sha256"] = hashlib.sha256(model.tex_data[start:start+length].tobytes()).hexdigest()
            textures.append(tex)
        out["textures"] = textures
        return out
    for i in range(model.nbody):
        if handbody(i): continue
        entry = _values(model, "body", "pos quat ipos iquat mass inertia gravcomp", i)
        entry["parent"] = bname(model.body_parentid[i])
        entry["mocap"] = bool(model.body_mocapid[i] >= 0)
        result["bodies"][bname(i)] = entry
    for i in range(model.njnt):
        if handbody(model.jnt_bodyid[i]): continue
        entry = _values(model, "jnt", "type pos axis range limited stiffness margin solref solimp", i)
        entry["body"] = bname(model.jnt_bodyid[i])
        dof = int(model.jnt_dofadr[i]); count = {0: 6, 1: 3, 2: 1, 3: 1}[int(model.jnt_type[i])]
        entry["dofs"] = _values(model, "dof", "damping armature frictionloss solref solimp", slice(dof, dof+count))
        result["joints"][model.joint(i).name] = entry
    for i in range(model.ngeom):
        body = model.geom_bodyid[i]
        if handbody(body): continue
        entry = _values(model, "geom", "type size pos quat friction solref solimp solmix margin gap condim contype conaffinity group priority rgba", i)
        entry["body"] = bname(body)
        entry["material"] = material(int(model.geom_matid[i]))
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[i]); start = int(model.mesh_vertadr[mid]); count = int(model.mesh_vertnum[mid])
            face = int(model.mesh_faceadr[mid]); nface = int(model.mesh_facenum[mid])
            entry["mesh"] = dict(vertices=digest(model.mesh_vert[start:start+count].tolist()),
                                 faces=digest(model.mesh_face[face:face+nface].tolist()),
                                 **_values(model, "mesh", "pos quat scale", mid))
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
            raise ValueError("Heightfield audit is not implemented; refusing incomplete comparison")
        ordinal = int(np.count_nonzero(model.geom_bodyid[:i] == body))
        name = model.geom(i).name or f"{bname(body)}:geom:{ordinal}"
        result["geoms"][name] = entry
    for i in range(model.nu):
        name = model.actuator(i).name
        if name.startswith(PREFIX): continue
        entry = _values(model, "actuator", "trntype dyntype gaintype biastype ctrlrange forcerange gear gainprm biasprm dynprm ctrllimited forcelimited", i)
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(f"Unrecognized non-hand transmission: {name}")
        entry["joint"] = model.joint(int(model.actuator_trnid[i, 0])).name
        result["actuators"][name] = entry
    for stem, category, count, fields in (
        ("site", "sites", model.nsite, "type pos quat size rgba group"),
        ("cam", "cameras", model.ncam, "pos quat fovy ipd resolution intrinsic sensorsize mode orthographic"),
        ("light", "lights", model.nlight, "pos dir diffuse ambient specular attenuation cutoff exponent castshadow type active mode intensity range bulbradius"),
    ):
        for i in range(count):
            body = getattr(model, stem+"_bodyid")[i]
            if handbody(body): continue
            accessor = getattr(model, "camera" if stem == "cam" else stem)
            entry = _values(model, stem, fields, i); entry["body"] = bname(body)
            if stem in ("cam","light"):
                target=int(getattr(model,stem+"_targetbodyid")[i])
                entry["target_body"]=bname(target) if target>=0 else None
            name = accessor(i).name or f"{bname(body)}:{stem}:{i}"
            result[category][name] = entry
    result["options"] = {name: np.asarray(getattr(model.opt, name)).tolist()
                         for name in dir(model.opt) if not name.startswith("_")}
    result["exclusions"] = sorted((bname(int(code)//65536), bname(int(code)%65536))
        for code in model.exclude_signature
        if not handbody(int(code)//65536) and not handbody(int(code)%65536))
    result["contact_pairs"]={}
    for i in range(model.npair):
        gids=(int(model.pair_geom1[i]),int(model.pair_geom2[i]))
        if any(handbody(model.geom_bodyid[g]) for g in gids):
            raise ValueError("Explicit hand contact pair requires a dedicated cross-embodiment review")
        names=[model.geom(g).name for g in gids]
        if not all(names):raise ValueError("Unnamed explicit contact-pair geometry")
        result["contact_pairs"]["|".join(sorted(names))]=_values(model,"pair","dim solref solreffriction solimp margin gap friction",i)
    result["equalities"]={}
    for i in range(model.neq):
        kind=int(model.eq_type[i]);ids=(int(model.eq_obj1id[i]),int(model.eq_obj2id[i]))
        if kind in (mujoco.mjtEq.mjEQ_CONNECT,mujoco.mjtEq.mjEQ_WELD):accessor=model.body
        elif kind==mujoco.mjtEq.mjEQ_JOINT:accessor=model.joint
        elif kind==mujoco.mjtEq.mjEQ_TENDON:accessor=model.tendon
        else:raise ValueError(f"Unsupported equality type {kind}")
        names=[accessor(idx).name if idx>=0 else None for idx in ids]
        if any(name and name.startswith(PREFIX) for name in names):continue
        entry=_values(model,"eq","type active0 solref solimp data",i);entry["objects"]=names
        result["equalities"][model.equality(i).name or "|".join(str(n) for n in names)]=entry
    for i in range(model.ntendon):
        if not model.tendon(i).name.startswith(PREFIX):
            raise ValueError("Non-hand tendon audit is not implemented; refusing incomplete comparison")
    return result


def differences(reference, actual, prefix=""):
    if isinstance(reference, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(reference) | set(actual)):
            path = f"{prefix}/{key}"
            if key not in reference or key not in actual: out.append(path + ": missing")
            else: out.extend(differences(reference[key], actual[key], path))
        return out
    if reference != actual: return [prefix]
    return []
