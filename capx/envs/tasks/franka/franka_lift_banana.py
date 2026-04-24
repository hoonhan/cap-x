from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Research 3 robot with API described below.
Goal: pick up the banana from the scene and lift it stably.
Use smooth and safe motions: approach from above, grasp, and lift without dropping.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""

# ORACLE_CODE is the built-in reference solution used by the environment as an expert policy.
# It is used for oracle rollout / debugging / evaluation baselines.
ORACLE_CODE = """
import numpy as np

# 1) Get a grasp pose for the banana
banana_grasp_pos, banana_grasp_quat = sample_grasp_pose("banana")

# 2) Ensure gripper is open before approach
open_gripper()

# 3) Approach from above for safer grasping
goto_pose(banana_grasp_pos, banana_grasp_quat, z_approach=0.12)

# 4) Move to exact grasp pose and grasp
goto_pose(banana_grasp_pos, banana_grasp_quat)
close_gripper()

# 5) Lift banana vertically
lift_offset = np.array([0.0, 0.0, 0.12])
lift_pos = banana_grasp_pos + lift_offset
goto_pose(lift_pos, banana_grasp_quat)
"""


class FrankaLiftBananaCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for lifting a banana with Franka Research 3."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "FrankaLiftBananaCodeEnv",
]
