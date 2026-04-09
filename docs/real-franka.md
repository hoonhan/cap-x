# Real Franka Panda QuickStart

Make sure you have [robots_realtime](https://github.com/uynitsuj/robots_realtime.git) cloned and have tested launching the real Franka Panda using the robots_realtime repo instructions. For example test with `configs/franka/franka_robotiq_viser_teleop.yaml` (uses a robotiq gripper interfacing with an RS485).

The default task configured in [`env_configs/real/real.yaml`](../env_configs/real/real.yaml) is **"pick up the red cube and lift it"**. Feel free to modify the `task_only_prompt` and `prompt` fields in that file to define your own task, e.g. another cool task that we have tried in real (and works!) is: **"stack these objects as high as possible"**

## Requirements

Beyond a Franka Panda robot arm, this workflow requires a **stereo camera that produces calibrated metric-scale depth maps** (e.g. a ZED stereo camera). The depth data is used by SAM3 + Contact-GraspNet to generate grasp poses in 3D. A monocular RGB-only camera may be insufficient.

## Camera Extrinsics Setup

The `robots_realtime` client config points to a camera extrinsics file that describes the camera's pose in the robot world frame. You must create one for your own setup.

**1. Create your extrinsics file** in `robots_realtime/configs/camera_extrinsics/`. Use the AutoLab ZED setup as a reference:

[`configs/camera_extrinsics/autolab_franka_zed_top.yaml`](https://github.com/uynitsuj/robots_realtime/blob/main/configs/camera_extrinsics/autolab_franka_zed_top.yaml)

```yaml
# Camera extrinsics for your Franka setup.
#
# All values are expressed in the robot world frame (base link origin).
#
# position: [x, y, z] in meters
# rpy_radians: [roll, pitch, yaw] in radians (applied in that order)
#
# For your own setup, create a new file in this directory, then point
# the ZedCamera config at it via the `extrinsics_file` field.

position: [1.007, 0.0, 0.29]
rpy_radians: [1.0472, 3.14159, -1.5708]
```

**2. Point the client config at your file.** In `robots_realtime/configs/franka/franka_robotiq_client.yaml` (or whichever client config you use), update the `extrinsics_file` field under the ZedCamera sensor node:

```yaml
extrinsics_file: "configs/camera_extrinsics/your_setup.yaml"
```

**3. Calibrate your extrinsics.** The `position` and `rpy_radians` values must match your physical camera mounting. Poor calibration will cause grasp poses to be offset from the real object locations.

**4. Then run the CaP-X real experiment configs.**
```bash
uv sync --active --extra contactgraspnet
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/real/real.yaml
```

Open up the interactive web UI at the provided port (defaulted to http://localhost:8200).

Once you see:
```
Waiting for observation from real environment...
Waiting for observation from real environment...
```
In a separate terminal with the [robots_realtime](https://github.com/uynitsuj/robots_realtime.git) repo set as the current directory, launch:
```bash
uv run rr-session configs/franka/franka_robotiq_client.yaml
```

---

## FR3 + Dual RealSense D405 (Minimal POC)

If your goal is a **minimal real-hardware proof of concept** (small primitive motions, no grasp planning stack), use:

```bash
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/real/fr3_d405_minimal.yaml
```

This config uses `FrankaRealMinimalApi`, which exposes small primitive actions:
- `move_to_joint_positions(joints)`
- `move_by_joint_delta(delta_joints)`
- `set_gripper_opening(fraction)`
- `open_gripper()`, `close_gripper()`
- `hold_position(duration_seconds=1.0, hz=25.0)`
- `get_camera_observation(camera_name=...)`

### Dual-camera extrinsics template

A starting template is provided at:

`env_configs/real/camera_extrinsics/fr3_dual_d405_example.yaml`

The template now includes your provided FR3 + D405 measured camera positions and look-at targets as a starting point:
- `camera_top`: `(1.35, 0.00, 0.53)` looking at `(0.20, 0.00, 0.00)`
- `camera_wrist_side`: `(0.50, 0.69, 0.50)` looking at `(0.50, 0.00, 0.10)`

Copy these values into your `robots_realtime` camera-extrinsics configuration.
The included `rpy_radians` are look-at-derived initial values and may need final refinement because camera-frame conventions can differ across sensor wrappers.

### Practical calibration guide (no extra calibration scripts required)

You mentioned `calibrate_camera` duplication is unnecessary; this workflow does **not** add any extra calibration tooling.

Use this lightweight process instead:
1. Start with the template extrinsics above.
2. Run `rr-session` + CaP-X minimal config and view the live feed(s).
3. Pick an object with known location in robot base frame.
4. Compare expected vs observed object location in each camera stream.
5. Tune each camera's `rpy_radians` (small increments, e.g. 0.02–0.05 rad) and then `position` (1–2 cm) until overlays / reprojections align.
6. Re-verify at multiple workspace points, not just one center point.

If you compare against GraspVLA configs, it can help as a **reference check** for layout conventions, but this CaP-X setup does not require installing GraspVLA or adding its dependencies.

### Multi-camera observation mapping

In the real low-level adapter:
- the first discovered `camera_*` stream is mapped to `robot0_robotview` (for compatibility),
- additional streams are preserved under their original keys (e.g. `camera_wrist_side`).

You can fetch any camera stream with:

```python
cam_top = get_camera_observation(\"robot0_robotview\")
cam_side = get_camera_observation(\"camera_wrist_side\")
```
