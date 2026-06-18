# Unitree RL Gym - Moving Plate Sim

This fork is only documenting the local MuJoCo moving-plate deploy changes.
For installation, training, export, Sim2Real, and the normal Unitree RL Gym workflow, refer to the main/upstream Unitree RL Gym README and the setup guide in [doc/setup_en.md](doc/setup_en.md).

## Run G1 on the Moving Plate

The G1 MuJoCo scene includes a HumanUp-style 6-DOF support plate. The normal floor is visual-only, so the robot contacts the plate instead of walking on the flat MuJoCo plane.
The visual checker ground is lowered 1 m below the plate so the moving support is easy to see in the viewer.

```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

The config lives at:

```text
deploy/deploy_mujoco/configs/g1.yaml
```

The scene with the plate lives at:

```text
resources/robots/g1_description/scene.xml
```

## Tune Viewer Camera

By default, the MuJoCo viewer tracks the robot base body, so the view follows the robot like the Digit viewer:

```yaml
viewer_camera:
  track_base: true
  distance: 3.0
  azimuth: -140.0
  elevation: -20.0
```

Set `track_base: false` to leave the viewer camera in the normal free-camera mode.

## Tune Plate Motion

Edit `plate_motion` in `deploy/deploy_mujoco/configs/g1.yaml`:

```yaml
plate_motion:
  enabled: true
  start_time: 0.0
  offset: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  amplitude: [0.10, 0.0, 0.0, 0.02618, 0.02618, 0.0]
  period: [4.5, 4.5, 5.0, 4.5, 4.5, 4.5]
  phase: [0.0, 0.785398, 0.349066, 0.523599, 0.0, 0.0]
  sagittal_acceleration:
    enabled: false
    start_time: 0.0
    initial_acceleration: 0.0
    target_acceleration: 0.1
    ramp_duration: 2.0
    initial_velocity: 0.0
```

Values are ordered as:

```text
[x, y, z, roll, pitch, yaw]
```

Linear channels are in meters, angular channels are in radians. Each channel follows:

```text
offset + amplitude * (1 - cos(2*pi*t/period + phase))
```

The optional `sagittal_acceleration` block adds an x-axis displacement to the plate command by integrating acceleration. During `ramp_duration`, acceleration ramps linearly from `initial_acceleration` to `target_acceleration`; after that, the plate continues with constant `target_acceleration`. Acceleration is in `m/s^2`, velocity is in `m/s`, and time is in seconds.

Useful tweaks:

- Set `enabled: false` to stop the plate servo command.
- Increase `amplitude[0]` or `amplitude[1]` for horizontal plate travel.
- Increase `amplitude[3]` or `amplitude[4]` for roll/pitch tilt.
- Increase `period` for slower motion; decrease it for faster motion.
- Set `start_time` to delay when the moving plate begins.
- Set `sagittal_acceleration.enabled: true` for ramped constant acceleration along plate x.

## Log and Plot Plate IMU

`deploy_mujoco.py` logs plate data to:

```text
deploy/deploy_mujoco/data
```

The first plotted signal is the plate IMU accelerometer:

```text
deploy/deploy_mujoco/data/plate_imu_acc.dat
```

The deploy run also logs COM velocity tracking data in the plate frame:

```text
deploy/deploy_mujoco/data/com_vel_plate.dat
deploy/deploy_mujoco/data/cmd_vel.dat
deploy/deploy_mujoco/data/com_vel_tracking_error.dat
```

Run the plotter after a deploy run:

```bash
python deploy/deploy_mujoco/plot_data.py
```

This saves:

```text
deploy/deploy_mujoco/data/plate_imu_acc.png
deploy/deploy_mujoco/data/com_velocity_tracking_error.png
```

Use `--plot com-velocity-tracking` to generate only the COM velocity tracking plot, and use `--show` to also open the matplotlib window.

## Notes

- `deploy_mujoco.py` reads robot joint state through the actuated joint addresses, so the extra plate joints do not enter the policy observation.
- The plate is controlled through applied joint forces using the same joint order as HumanUp: `plate_x`, `plate_y`, `plate_z`, `plate_roll`, `plate_pitch`, `plate_yaw`.
- The default G1 policy path remains `deploy/pre_train/g1/motion.pt`; update `policy_path` in `g1.yaml` to use a custom exported policy.
