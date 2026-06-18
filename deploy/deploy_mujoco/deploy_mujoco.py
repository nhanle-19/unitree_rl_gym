import time
from pathlib import Path

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


PLATE_JOINT_ORDER = ("plate_x", "plate_y", "plate_z", "plate_roll", "plate_pitch", "plate_yaw")
PLATE_GEOM_NAME = "plate_collision_0"
PLATE_IMU_ACCEL_SENSOR = "plate_imu-accelerometer"
DATA_DIR = Path(__file__).resolve().parent / "data"
PLATE_SERVO_KP = np.array(
    [1_000_000.0, 1_000_000.0, 1_000_000.0, 3_000_000.0, 3_000_000.0, 3_000_000.0]
)
PLATE_SERVO_KV = np.array([10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0])
PLATE_SERVO_FORCE_LIMIT = np.array(
    [10_000_000.0, 10_000_000.0, 10_000_000.0, 10_000_000.0, 10_000_000.0, 10_000_000.0]
)


def get_actuated_joint_addresses(model, num_actions):
    if model.nu < num_actions:
        raise RuntimeError(f"Model has {model.nu} actuators, but config expects {num_actions} actions")

    qpos_addr = []
    dof_addr = []
    for actuator_id in range(num_actions):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise RuntimeError(f"Actuator {actuator_id} is not attached to a joint")
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        if qadr < 0 or dadr < 0:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise RuntimeError(f"Actuated joint {joint_name!r} does not expose qpos/qvel addresses")
        qpos_addr.append(qadr)
        dof_addr.append(dadr)

    return np.array(qpos_addr, dtype=np.int32), np.array(dof_addr, dtype=np.int32)


def load_plate_motion(config):
    motion = config.get("plate_motion", {})
    enabled = bool(motion.get("enabled", False))

    def plate_array(key, default):
        value = motion.get(key, default)
        array = np.array(value, dtype=np.float64)
        if array.shape != (len(PLATE_JOINT_ORDER),):
            raise ValueError(f"plate_motion.{key} must have {len(PLATE_JOINT_ORDER)} values")
        return array

    period = plate_array("period", [4.5, 4.5, 5.0, 4.5, 4.5, 4.5])
    if np.any(period <= 0.0):
        raise ValueError("plate_motion.period values must be positive")

    return {
        "enabled": enabled,
        "start_time": float(motion.get("start_time", 0.0)),
        "offset": plate_array("offset", [0.0] * len(PLATE_JOINT_ORDER)),
        "amplitude": plate_array("amplitude", [0.0] * len(PLATE_JOINT_ORDER)),
        "period": period,
        "phase": plate_array("phase", [0.0] * len(PLATE_JOINT_ORDER)),
    }


def get_plate_command(t, plate_motion):
    t_eff = max(0.0, float(t) - plate_motion["start_time"])
    phase = 2.0 * np.pi * t_eff / plate_motion["period"] + plate_motion["phase"]
    return plate_motion["offset"] + plate_motion["amplitude"] * (1.0 - np.cos(phase))


def get_plate_joint_addresses(model, require_plate):
    qpos_addr = []
    dof_addr = []
    missing = []
    for name in PLATE_JOINT_ORDER:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            missing.append(name)
            continue
        qpos_addr.append(int(model.jnt_qposadr[joint_id]))
        dof_addr.append(int(model.jnt_dofadr[joint_id]))

    if missing:
        if require_plate:
            raise RuntimeError(f"Plate motion is enabled, but these plate joints are missing: {missing}")
        return None

    return np.array(qpos_addr, dtype=np.int32), np.array(dof_addr, dtype=np.int32)


def configure_plate_model(model, plate_motion):
    plate_addresses = get_plate_joint_addresses(model, plate_motion["enabled"])
    if plate_addresses is None:
        return None

    plate_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PLATE_GEOM_NAME)
    if plate_geom_id == -1 and plate_motion["enabled"]:
        raise RuntimeError(f"Plate motion is enabled, but geom {PLATE_GEOM_NAME!r} is missing")
    if plate_geom_id != -1:
        model.geom_friction[plate_geom_id] = [2.5, 0.005, 0.0001]
        model.geom_condim[plate_geom_id] = 4
        model.geom_contype[plate_geom_id] = 128
        model.geom_conaffinity[plate_geom_id] = 5
        model.geom_priority[plate_geom_id] = 100
        model.geom_margin[plate_geom_id] = 0.002
        model.geom_solref[plate_geom_id] = [0.002, 1.0]
        model.geom_solimp[plate_geom_id] = [0.95, 0.99, 0.001, 0.5, 2.0]

    _, plate_dof_addr = plate_addresses
    model.dof_armature[plate_dof_addr] = 10.0
    model.dof_damping[plate_dof_addr] = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    model.dof_frictionloss[plate_dof_addr] = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    return plate_addresses


def apply_plate_state(data, plate_addresses, plate_command):
    plate_qpos_addr, plate_dof_addr = plate_addresses
    data.qpos[plate_qpos_addr] = plate_command
    data.qvel[plate_dof_addr] = 0.0


def apply_plate_position_control(data, plate_addresses, plate_command):
    plate_qpos_addr, plate_dof_addr = plate_addresses
    force = (
        PLATE_SERVO_KP * (plate_command - data.qpos[plate_qpos_addr])
        - PLATE_SERVO_KV * data.qvel[plate_dof_addr]
    )
    data.qfrc_applied[plate_dof_addr] = np.clip(force, -PLATE_SERVO_FORCE_LIMIT, PLATE_SERVO_FORCE_LIMIT)


class PlateDataLogger:
    def __init__(self, model, plate_addresses, output_dir=DATA_DIR):
        self.output_dir = Path(output_dir)
        self.plate_addresses = plate_addresses
        self.t = []
        self.plate_imu_acc = []
        self.drs_des = []
        self.drs_act = []

        self.plate_imu_acc_sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, PLATE_IMU_ACCEL_SENSOR
        )
        if self.plate_imu_acc_sid >= 0:
            self.plate_imu_acc_adr = int(model.sensor_adr[self.plate_imu_acc_sid])
            self.plate_imu_acc_dim = int(model.sensor_dim[self.plate_imu_acc_sid])
        else:
            self.plate_imu_acc_adr = -1
            self.plate_imu_acc_dim = 0

    def update(self, data, t, plate_command=None):
        self.t.append([float(t)])
        self.plate_imu_acc.append(self._read_plate_imu_acc(data))

        if plate_command is None:
            plate_command = np.zeros(len(PLATE_JOINT_ORDER), dtype=np.float32)
        self.drs_des.append(np.asarray(plate_command, dtype=np.float32).reshape(len(PLATE_JOINT_ORDER)))

        if self.plate_addresses is None:
            plate_state = np.zeros(len(PLATE_JOINT_ORDER), dtype=np.float32)
        else:
            plate_qpos_addr, _ = self.plate_addresses
            plate_state = data.qpos[plate_qpos_addr].astype(np.float32)
        self.drs_act.append(plate_state)

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_array("t", self.t, 1)
        self._save_array("plate_imu_acc", self.plate_imu_acc, 3)
        self._save_array("drs_des", self.drs_des, len(PLATE_JOINT_ORDER))
        self._save_array("drs_act", self.drs_act, len(PLATE_JOINT_ORDER))
        print(f"Saved MuJoCo deploy data to {self.output_dir}")

    def _read_plate_imu_acc(self, data):
        if self.plate_imu_acc_sid < 0 or self.plate_imu_acc_dim < 3:
            return np.zeros(3, dtype=np.float32)
        acc = data.sensordata[self.plate_imu_acc_adr : self.plate_imu_acc_adr + 3]
        return np.asarray(acc, dtype=np.float32).reshape(3)

    def _save_array(self, name, values, width):
        if values:
            array = np.asarray(values, dtype=np.float32).reshape(-1, width)
        else:
            array = np.empty((0, width), dtype=np.float32)
        np.savetxt(self.output_dir / f"{name}.dat", array)


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        plate_motion = load_plate_motion(config)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    robot_qpos_addr, robot_dof_addr = get_actuated_joint_addresses(m, num_actions)
    plate_addresses = configure_plate_model(m, plate_motion)
    if plate_motion["enabled"]:
        apply_plate_state(d, plate_addresses, get_plate_command(0.0, plate_motion))
        mujoco.mj_forward(m, d)
    data_logger = PlateDataLogger(m, plate_addresses)

    # load policy
    policy = torch.jit.load(policy_path)

    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Close the viewer automatically after simulation_duration wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < simulation_duration:
                step_start = time.time()
                sim_time = counter * simulation_dt
                if plate_motion["enabled"]:
                    plate_command = get_plate_command(sim_time, plate_motion)
                    apply_plate_position_control(d, plate_addresses, plate_command)
                else:
                    plate_command = np.zeros(len(PLATE_JOINT_ORDER), dtype=np.float64)

                tau = pd_control(
                    target_dof_pos,
                    d.qpos[robot_qpos_addr],
                    kps,
                    np.zeros_like(kds),
                    d.qvel[robot_dof_addr],
                    kds,
                )
                d.ctrl[:num_actions] = tau
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)
                data_logger.update(d, d.time, plate_command)

                counter += 1
                if counter % control_decimation == 0:
                    # Apply control signal here.

                    # create observation
                    qj = d.qpos[robot_qpos_addr]
                    dqj = d.qvel[robot_dof_addr]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * ang_vel_scale

                    period = 0.8
                    count = counter * simulation_dt
                    phase = count % period / period
                    sin_phase = np.sin(2 * np.pi * phase)
                    cos_phase = np.cos(2 * np.pi * phase)

                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9 : 9 + num_actions] = qj
                    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                    obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    # policy inference
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    # transform action to target_dof_pos
                    target_dof_pos = action * action_scale + default_angles

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    finally:
        data_logger.save()
