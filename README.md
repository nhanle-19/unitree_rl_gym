# Unitree RL Gym - Moving Plate Sim

This fork is only documenting the local MuJoCo moving-plate deploy changes.
For installation, training, export, Sim2Real, and the normal Unitree RL Gym workflow, refer to the main/upstream Unitree RL Gym README and the setup guide in [doc/setup_en.md](doc/setup_en.md).

## Run G1 on the Moving Plate

The G1 MuJoCo scene includes a HumanUp-style 6-DOF support plate. The normal floor is visual-only, so the robot contacts the plate instead of walking on the flat MuJoCo plane.
The visual checker ground is lowered below the plate so the moving support is easy to see in the viewer.

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
```

Values are ordered as:

```text
[x, y, z, roll, pitch, yaw]
```

Linear channels are in meters, angular channels are in radians. Each channel follows:

```text
offset + amplitude * (1 - cos(2*pi*t/period + phase))
```

Useful tweaks:

- Set `enabled: false` to stop the plate servo command.
- Increase `amplitude[0]` or `amplitude[1]` for horizontal plate travel.
- Increase `amplitude[3]` or `amplitude[4]` for roll/pitch tilt.
- Increase `period` for slower motion; decrease it for faster motion.
- Set `start_time` to delay when the moving plate begins.

## Notes

- `deploy_mujoco.py` reads robot joint state through the actuated joint addresses, so the extra plate joints do not enter the policy observation.
- The plate is controlled through applied joint forces using the same joint order as HumanUp: `plate_x`, `plate_y`, `plate_z`, `plate_roll`, `plate_pitch`, `plate_yaw`.
- The default G1 policy path remains `deploy/pre_train/g1/motion.pt`; update `policy_path` in `g1.yaml` to use a custom exported policy.
