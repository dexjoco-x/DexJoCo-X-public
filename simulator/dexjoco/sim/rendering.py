from __future__ import annotations

from typing import Optional

import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import (
    MujocoRenderer as GymnasiumMujocoRenderer,
)


class MujocoRenderer(GymnasiumMujocoRenderer):
    """Compatibility wrapper for Gymnasium 0.29 and 1.x MuJoCo renderers."""

    def __init__(self, model, data, *args, width=None, height=None, **kwargs):
        if width is None:
            width = int(model.vis.global_.offwidth)
        if height is None:
            height = int(model.vis.global_.offheight)
        super().__init__(model, data, *args, width=width, height=height, **kwargs)
        self._reuse_rgb_array = False
        self._rgb_array_cache = {}

    def set_reuse_rgb_array(self, reuse: bool) -> None:
        """Reuse the last frame while physics advances between policy replans."""
        self._reuse_rgb_array = bool(reuse)

    def render(
        self,
        render_mode: Optional[str],
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        cache_key = (camera_id, camera_name)
        if (
            render_mode == "rgb_array"
            and self._reuse_rgb_array
            and cache_key in self._rgb_array_cache
        ):
            return self._rgb_array_cache[cache_key]

        try:
            frame = super().render(
                render_mode=render_mode,
                camera_id=camera_id,
                camera_name=camera_name,
            )
            if render_mode == "rgb_array":
                self._rgb_array_cache[cache_key] = frame
            return frame
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise

        if camera_id is not None and camera_name is not None:
            raise ValueError("camera_id and camera_name cannot both be specified.")

        if camera_name is not None:
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )

        if camera_id is not None:
            self.camera_id = camera_id

        frame = super().render(render_mode=render_mode)
        if render_mode == "rgb_array":
            self._rgb_array_cache[cache_key] = frame
        return frame
