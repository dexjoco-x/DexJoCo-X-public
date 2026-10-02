"""Rigid mounting checks, not a certification of a real manufactured adapter."""
from __future__ import annotations
import mujoco
import numpy as np


def audit_wrist(model, side, dual=True):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    prefix = "benchhand_" + side + "_"
    root = model.body("attachment_" + side if dual else "attachment").id
    arm = int(model.body_parentid[root])
    weld = int(model.body_weldid[root])
    bodies = [i for i in range(model.nbody) if model.body(i).name.startswith(prefix)]
    rigid = [i for i in bodies if model.body_weldid[i] == weld]
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in rigid]
    visible = [g for g in geoms if model.geom_group[g] < 3 and model.geom_rgba[g, 3] > 0]
    arm_visible = [g for g in range(model.ngeom)
                   if model.geom_bodyid[g] == arm and model.geom_group[g] < 3]
    issues = []
    for body in bodies:
        current = body
        while current != root and current != 0:
            current = int(model.body_parentid[current])
        if current != root: issues.append("hand body outside attachment subtree")
    joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] in bodies]
    if any(model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL)
           for j in joints):
        issues.append("unexpected free/ball joint in native hand subtree")
    mass = float(model.body_mass[rigid].sum())
    if mass <= 0: issues.append("massless rigid hand base")
    distance = min((float(mujoco.mj_geomDistance(model, data, a, b, 1., None))
                    for a in arm_visible for b in visible), default=float("inf"))
    # CAD polygon discretization gives Allegro .70 mm and Shadow .34 mm.
    # Signed convex distances can report overlap inside hollow housings: they
    # are a gap screen, not a mesh-intersection or machining-tolerance test.
    if distance > .001: issues.append("arm-to-base visible gap exceeds 1 mm")
    adapters = []
    for visual in visible:
        name = model.geom(visual).name
        if "adapter" not in name or "visual" not in name: continue
        matches = []
        for candidate in geoms:
            if not (model.geom_contype[candidate] or model.geom_conaffinity[candidate]): continue
            if model.geom_bodyid[candidate] != model.geom_bodyid[visual]: continue
            if model.geom_type[candidate] != model.geom_type[visual]: continue
            if all(np.array_equal(getattr(model, key)[candidate], getattr(model, key)[visual])
                   for key in ("geom_pos", "geom_quat", "geom_size")):
                matches.append(candidate)
        if not matches: issues.append(name + " has no matching collision geometry")
        palms = [g for g in visible if "adapter" not in model.geom(g).name
                 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        adapters.append(dict(visual=name, collision=[model.geom(g).name for g in matches],
                             nearest_palm_convex_distance_m=min(
                                 (float(mujoco.mj_geomDistance(model, data, visual, g, 1., None))
                                  for g in palms), default=None)))
    native_wrist = [model.joint(j).name for j in joints if "WRJ" in model.joint(j).name]
    if any(w.number for w in data.warning): issues.append("MuJoCo forward warning")
    return dict(passed=not issues, issues=issues, side=side,
                rigid_body_names=[model.body(i).name for i in rigid], rigid_base_mass_kg=mass,
                arm_base_convex_gap_m=distance, adapters=adapters,
                native_articulated_wrist_joints=native_wrist,
                hardware_mount_certified=False,
                limits="Gap/rigidity/collision checks plus separate visual review; "
                       "convex overlap is not a true visual penetration measurement.")
