"""Per-step depth capture for Dexjoco envs, and helpers to write depth
videos alongside the RGB captures.

The depth image is what MuJoCo calls the "linearised z-buffer": the
distance from the camera plane to the closest surface, in metres. It is
not Euclidean distance to the optical centre.

Each env's `_compute_observation()` already calls `render()` to produce
RGB frames; this module mirrors that call but with `render_mode=
"depth_array"` so the camera-id ordering matches the dict ordering of
`obs["images"]` exactly.

Writing:
- `<name>_depth.npz`: a compressed archive with a single float32 array
  named "depth" of shape (T, H, W), in metres. Use this for training or
  any consumer that needs the true depth.
- `<name>_depth.mp4`: an 8-bit gray-scale H.264 preview, normalised to
  the per-clip min/max of valid depth pixels. Lossy.

Invalid pixels (depth equal to mujoco's far plane, returned when no
geom is hit) are kept as float `inf` in the npz so consumers can mask
them, and clipped out of the normalisation when building the mp4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


AUTOCOLLECT_CAMERA_NAMES = {
    "fixed_front": "autocollect_fixed_front",
    "top": "autocollect_top",
}


def _env_viewer_and_camera_ids(raw_env, image_keys):
    """Return (viewer, [camera_id_per_image_key]) for the given env.

    All envs except water_plant store the renderer at `_viewer` and the
    ordered camera-id tuple at `camera_id`. water_plant uses `_mj_viewer`
    and exposes the two ids individually. We probe in that order.
    """
    viewer = getattr(raw_env, "_viewer", None)
    if viewer is None:
        viewer = getattr(raw_env, "_mj_viewer", None)
    if viewer is None:
        return None, None

    cam_ids = getattr(raw_env, "camera_id", None)
    if cam_ids is not None:
        cam_ids = tuple(cam_ids)
        if len(cam_ids) == len(image_keys) and not any(
            key in AUTOCOLLECT_CAMERA_NAMES for key in image_keys
        ):
            return viewer, cam_ids

    named_ids = {
        "front": getattr(raw_env, "_front_camera_id", None),
        "random_camera": getattr(raw_env, "_front_camera_id", None),
        "ego": getattr(raw_env, "_front_camera_id", None),
        "ego_left": getattr(raw_env, "_left_camera_id", None),
        "ego_right": getattr(raw_env, "_right_camera_id", None),
        "wrist": getattr(raw_env, "_wrist_camera_id", None),
        "wrist_left": getattr(raw_env, "_wrist_left_camera_id", None),
        "wrist_right": getattr(raw_env, "_wrist_right_camera_id", None),
    }
    # Resolve these by explicit model names, never by tuple length/order. A
    # missing extra camera must not silently become a legacy or free camera.
    for key in image_keys:
        if key in AUTOCOLLECT_CAMERA_NAMES:
            model = getattr(raw_env, "_model", None)
            if model is None:
                model = getattr(raw_env, "model", None)
            try:
                camera_id = int(model.camera(AUTOCOLLECT_CAMERA_NAMES[key]).id)
            except (AttributeError, KeyError, ValueError):
                return viewer, None
            if camera_id < 0:
                return viewer, None
            named_ids[key] = camera_id
    # hammer_nail observations use (front, ego_left, ego_right, wrist).
    # Resolve named subsets without changing the legacy full-tuple fast path.
    if cam_ids is not None and len(cam_ids) >= 4:
        ordered = {
            "front": cam_ids[0],
            "random_camera": cam_ids[0],
            "ego": cam_ids[0],
            "ego_left": cam_ids[1],
            "ego_right": cam_ids[2],
            "wrist": cam_ids[3],
        }
        for key, camera_id in ordered.items():
            if named_ids.get(key) is None:
                named_ids[key] = camera_id
    selected = [named_ids.get(key) for key in image_keys]
    if all(cid is not None for cid in selected):
        return viewer, tuple(selected)

    if any(key in AUTOCOLLECT_CAMERA_NAMES for key in image_keys):
        return viewer, None

    wrist = getattr(raw_env, "_wrist_camera_id", None)
    front = getattr(raw_env, "_front_camera_id", None)
    if wrist is None or front is None or len(image_keys) != 2:
        return viewer, None
    # water_plant render() returns [wrist, front/random_camera].
    return viewer, (wrist, front)


def _linearise_depth(zbuf, znear, zfar):
    """Convert mujoco's nonlinear depth buffer (`mjr_readPixels` returns the
    OpenGL z-buffer in [0, 1]) into linear depth in metres along the camera
    ray axis. Pixels at the far plane (z == 1) become `inf`."""
    z = np.asarray(zbuf, dtype=np.float32)
    denom = zfar - z * (zfar - znear)
    out = np.full_like(z, np.inf, dtype=np.float32)
    mask = denom > 0
    out[mask] = (znear * zfar / denom)[mask]
    return out


def collect_depth_frames(env, image_keys):
    """Render a depth_array for each RGB camera key in `image_keys`.

    Returns a dict mapping the same key (e.g. "wrist") to a float32
    HxW depth image in metres, or an empty dict if depth capture is
    not available for this env.
    """
    raw = env.unwrapped
    viewer, cam_ids = _env_viewer_and_camera_ids(raw, image_keys)
    if viewer is None or cam_ids is None:
        return {}
    extent = float(raw._model.stat.extent)
    znear = float(raw._model.vis.map.znear) * extent
    zfar = float(raw._model.vis.map.zfar) * extent
    out = {}
    for key, cid in zip(image_keys, cam_ids):
        zbuf = viewer.render(render_mode="depth_array", camera_id=cid)
        out[key] = _linearise_depth(zbuf, znear, zfar)
    return out


def collect_segmentation_frames(env, image_keys):
    """Render MuJoCo object-type/object-id images aligned with RGB-D frames.

    Each returned HxWx2 int32 array stores ``(mjtObj, object_id)``.  Rendering
    goes directly through Gymnasium's off-screen viewer because its public
    ``MujocoRenderer.render`` wrapper does not expose the segmentation flag.
    """
    raw = env.unwrapped
    renderer, cam_ids = _env_viewer_and_camera_ids(raw, image_keys)
    if renderer is None or cam_ids is None:
        return {}
    viewer = renderer._get_viewer("rgb_array")
    out = {}
    for key, cid in zip(image_keys, cam_ids):
        try:
            out[key] = viewer.render(
                render_mode="rgb_array",
                camera_id=int(cid),
                segmentation=True,
            )
        except OverflowError:
            # Gymnasium 1.0 performs ID-color decoding in uint8.  NumPy 2.x
            # correctly rejects multiplication by 256/65536 as overflow,
            # while NumPy 1.x silently promoted it.  Render the same MuJoCo
            # segmentation pass and explicitly upcast before decoding so the
            # data contract is stable across both environments.
            out[key] = _render_segmentation_numpy2_compatible(viewer, int(cid))
    return out


def _render_segmentation_numpy2_compatible(viewer, camera_id: int) -> np.ndarray:
    import mujoco

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = int(camera_id)
    mujoco.mjv_updateScene(
        viewer.model,
        viewer.data,
        viewer.vopt,
        viewer.pert,
        viewer.cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        viewer.scn,
    )
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
    try:
        mujoco.mjr_render(viewer.viewport, viewer.scn, viewer.con)
        rgb = np.zeros(
            3 * viewer.viewport.width * viewer.viewport.height, dtype=np.uint8
        )
        depth = np.zeros(
            viewer.viewport.width * viewer.viewport.height, dtype=np.float32
        )
        mujoco.mjr_readPixels(rgb, depth, viewer.viewport, viewer.con)
    finally:
        viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        viewer.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

    rgb = rgb.reshape(viewer.viewport.height, viewer.viewport.width, 3)
    rgb32 = rgb.astype(np.int32)
    seg_image = rgb32[..., 0] + rgb32[..., 1] * 256 + rgb32[..., 2] * 65536
    seg_image[seg_image >= (viewer.scn.ngeom + 1)] = 0
    seg_ids = np.full((viewer.scn.ngeom + 1, 2), -1, dtype=np.int32)
    for index in range(viewer.scn.ngeom):
        geom = viewer.scn.geoms[index]
        if geom.segid != -1:
            seg_ids[geom.segid + 1, 0] = geom.objtype
            seg_ids[geom.segid + 1, 1] = geom.objid
    return seg_ids[seg_image][::-1, :, :]


def _normalise_depth_for_preview(depth_frames):
    """Stack and 8-bit-quantise a list of float depth frames for an mp4
    preview. The per-clip min/max are taken over finite pixels so very
    distant 'sky' pixels don't crush all the detail to a single bin."""
    stack = np.stack(depth_frames, axis=0)
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(finite.min())
        hi = float(finite.max())
        if hi <= lo:
            hi = lo + 1e-6
    clipped = np.clip(stack, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    norm[~np.isfinite(stack)] = 0.0
    gray = (norm * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1), lo, hi


def write_depth_outputs(depth_frames_per_camera, videos_dir, video_fps):
    from dexjoco.data.video_writer import Mp4VideoWriter

    """For each camera, write `<cam>_depth.npz` (float32 metres) and
    `<cam>_depth.mp4` (gray H.264 preview, normalised per clip).

    `depth_frames_per_camera` is a dict mapping camera key to a list of
    per-step depth frames. The lists may be empty (camera was never
    rendered) - in which case nothing is written for that key.
    """
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for cam_key, frames in depth_frames_per_camera.items():
        if not frames:
            continue
        depth_stack = np.stack(frames, axis=0).astype(np.float32)
        npz_path = videos_dir / f"{cam_key}_depth.npz"
        np.savez_compressed(npz_path, depth=depth_stack)
        saved.append(str(npz_path))

        preview, lo, hi = _normalise_depth_for_preview(frames)
        mp4_path = videos_dir / f"{cam_key}_depth.mp4"
        writer = Mp4VideoWriter.create_h264(
            fps=video_fps,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=21,
            thread_type="FRAME",
            thread_count=2,
        )
        writer.start(str(mp4_path))
        for frame in preview:
            writer.write_frame(frame)
        writer.stop()
        saved.append(f"{mp4_path} (depth range [{lo:.3f}, {hi:.3f}] m)")
    return saved
