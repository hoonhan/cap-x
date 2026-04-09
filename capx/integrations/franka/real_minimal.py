from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


class FrankaRealMinimalApi(ApiBase):
    """Minimal low-level control API for real Franka/FR3 bringup.

    This API intentionally avoids heavyweight vision / grasp dependencies and
    exposes only a small set of primitive motion actions suitable for proof-of-
    concept experiments on real hardware.
    """

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_camera_observation": self.get_camera_observation,
            "move_to_joint_positions": self.move_to_joint_positions,
            "move_by_joint_delta": self.move_by_joint_delta,
            "set_gripper_opening": self.set_gripper_opening,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "hold_position": self.hold_position,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return the most recent full observation dictionary from the real env."""
        return self._env.get_observation()

    def get_camera_observation(self, camera_name: str = "robot0_robotview") -> dict[str, np.ndarray]:
        """Get RGB-D camera data for a specific camera stream.

        Args:
            camera_name: Camera key in the observation dict. Default is
                ``robot0_robotview``.

        Returns:
            Dict containing at least ``rgb`` when available, and optionally
            ``depth``, ``intrinsics``, ``pose``, and ``pose_mat``.

        Raises:
            KeyError: If the requested camera does not exist in the observation.
        """
        obs = self._env.get_observation()
        if camera_name not in obs:
            available = sorted([k for k, v in obs.items() if isinstance(v, dict) and "images" in v])
            raise KeyError(
                f"Camera '{camera_name}' not present. Available cameras: {available}"
            )

        cam = obs[camera_name]
        images = cam.get("images", {})

        out: dict[str, np.ndarray] = {}
        for key in ("rgb", "depth"):
            if key in images and images[key] is not None:
                out[key] = np.asarray(images[key])

        for key in ("intrinsics", "pose", "pose_mat"):
            if key in cam and cam[key] is not None:
                out[key] = np.asarray(cam[key])

        return out

    def move_to_joint_positions(self, joints: list[float] | np.ndarray) -> None:
        """Move the arm to a target 7-DoF joint configuration.

        Args:
            joints: Target arm joints in radians, length 7.
        """
        joints_np = np.asarray(joints, dtype=np.float64).reshape(-1)
        if joints_np.shape[0] != 7:
            raise ValueError(f"Expected 7 joints, got shape {joints_np.shape}")
        self._env.move_to_joints_blocking(joints_np)

    def move_by_joint_delta(self, delta_joints: list[float] | np.ndarray) -> np.ndarray:
        """Move by a relative joint delta from current measured arm joints.

        Args:
            delta_joints: Joint increments in radians, length 7.

        Returns:
            The commanded absolute target joint positions.
        """
        obs = self._env.get_observation()
        if "robot_joint_pos" not in obs:
            raise RuntimeError("Observation does not contain 'robot_joint_pos'")

        current = np.asarray(obs["robot_joint_pos"], dtype=np.float64).reshape(-1)
        if current.shape[0] < 7:
            raise RuntimeError(f"Expected at least 7 joint values, got {current.shape}")

        delta_np = np.asarray(delta_joints, dtype=np.float64).reshape(-1)
        if delta_np.shape[0] != 7:
            raise ValueError(f"Expected 7 joint deltas, got shape {delta_np.shape}")

        target = current[:7] + delta_np
        self._env.move_to_joints_blocking(target)
        return target

    def set_gripper_opening(self, fraction: float) -> None:
        """Set gripper opening fraction.

        Args:
            fraction: 0.0 (fully closed) to 1.0 (fully open).
        """
        self._env._set_gripper(float(np.clip(fraction, 0.0, 1.0)))
        self._env._step_once()

    def open_gripper(self) -> None:
        """Fully open the gripper."""
        self.set_gripper_opening(1.0)

    def close_gripper(self) -> None:
        """Fully close the gripper."""
        self.set_gripper_opening(0.0)

    def hold_position(self, duration_seconds: float = 1.0, hz: float = 25.0) -> None:
        """Keep republishing the current command for a short duration.

        Useful for stabilizing communication with real-time clients that expect
        periodic command updates.

        Args:
            duration_seconds: Hold duration in seconds.
            hz: Publish rate during hold.
        """
        steps = max(1, int(duration_seconds * hz))
        for _ in range(steps):
            self._env._step_once()
