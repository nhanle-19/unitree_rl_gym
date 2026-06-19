from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"


def has_dat(name):
    return (DATA_DIR / f"{name}.dat").is_file()


def load_dat(name, width):
    path = DATA_DIR / f"{name}.dat"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run deploy_mujoco.py first.")

    data = np.loadtxt(path, dtype=np.float32)
    if data.size == 0:
        return np.empty((0, width), dtype=np.float32)
    return np.asarray(data, dtype=np.float32).reshape(-1, width)


def plot_plate_imu(save=True, show=False):
    t = load_dat("t", 1)[:, 0]
    acc = load_dat("plate_imu_acc", 3)
    n = min(t.shape[0], acc.shape[0])
    t = t[:n]
    acc = acc[:n]

    fig, axes = plt.subplots(3, 1, sharex=True, num="Plate IMU acceleration")
    labels = ("acc x", "acc y", "acc z")
    for idx, ax in enumerate(axes):
        ax.plot(t, acc[:, idx], "-")
        ax.set_ylabel(f"{labels[idx]} [m/s^2]")
        ax.grid(True, "both", "both")
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()

    if save:
        out = DATA_DIR / "plate_imu_acc.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_com_velocity_tracking_error(save=True, show=False):
    t = load_dat("t", 1)[:, 0]
    com_vel = load_dat("com_vel_plate", 3)
    cmd_vel = load_dat("cmd_vel", 2)
    error = load_dat("com_vel_tracking_error", 3)
    n = min(t.shape[0], com_vel.shape[0], cmd_vel.shape[0], error.shape[0])
    t = t[:n]
    com_vel = com_vel[:n]
    cmd_vel = cmd_vel[:n]
    error = error[:n]

    fig, axes = plt.subplots(3, 1, sharex=True, num="COM velocity tracking error")
    components = (("x", 0), ("y", 1))
    for ax, (label, idx) in zip(axes[:2], components):
        ax.plot(t, cmd_vel[:, idx], "--", label=f"cmd {label}")
        ax.plot(t, com_vel[:, idx], "-", label=f"COM {label} in plate frame")
        ax.plot(t, error[:, idx], ":", label=f"error {label}")
        ax.set_ylabel(f"{label} [m/s]")
        ax.grid(True, "both", "both")
        ax.legend(loc="best")

    axes[2].plot(t, error[:, 2], "-", label="xy error norm")
    axes[2].set_ylabel("error norm [m/s]")
    axes[2].set_xlabel("t [s]")
    axes[2].grid(True, "both", "both")
    axes[2].legend(loc="best")
    fig.tight_layout()

    if save:
        out = DATA_DIR / "com_velocity_tracking_error.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_com_tracking_map(save=True, show=False):
    com_xy = load_dat("com_pos_plate_xy", 2)
    cmd_xy = load_dat("cmd_pos_xy", 2)
    n = min(com_xy.shape[0], cmd_xy.shape[0])
    com_xy = com_xy[:n]
    cmd_xy = cmd_xy[:n]

    fig, ax = plt.subplots(num="COM tracking map")
    ax.plot(cmd_xy[:, 0], cmd_xy[:, 1], "--", label="cmd xy")
    ax.plot(com_xy[:, 0], com_xy[:, 1], "-", label="COM xy in plate frame")
    if n > 0:
        ax.plot(com_xy[0, 0], com_xy[0, 1], "go", label="start")
        ax.plot(com_xy[-1, 0], com_xy[-1, 1], "ro", label="end")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, "both", "both")
    ax.legend(loc="best")
    fig.tight_layout()

    if save:
        out = DATA_DIR / "com_tracking_map_xy.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ankle_torque(save=True, show=False):
    t = load_dat("t", 1)[:, 0]
    adaptive_tau = load_dat("adaptive_ankle_torque", 4)
    applied_tau = load_dat("applied_ankle_torque", 4)
    n = min(t.shape[0], adaptive_tau.shape[0], applied_tau.shape[0])
    t = t[:n]
    adaptive_tau = adaptive_tau[:n]
    applied_tau = applied_tau[:n]

    fig, axes = plt.subplots(4, 1, sharex=True, num="Ankle torque")
    labels = (
        "left ankle pitch",
        "left ankle roll",
        "right ankle pitch",
        "right ankle roll",
    )
    for idx, ax in enumerate(axes):
        ax.plot(t, applied_tau[:, idx], "-", label="applied total")
        ax.plot(t, adaptive_tau[:, idx], "--", label="adaptive overlay")
        ax.set_ylabel(f"{labels[idx]} [Nm]")
        ax.grid(True, "both", "both")
        ax.legend(loc="best")
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()

    if save:
        out = DATA_DIR / "ankle_torque.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    plots = (
        ("plate IMU plot", ("t", "plate_imu_acc"), plot_plate_imu),
        (
            "COM velocity tracking plot",
            ("t", "com_vel_plate", "cmd_vel", "com_vel_tracking_error"),
            plot_com_velocity_tracking_error,
        ),
        ("COM tracking map", ("com_pos_plate_xy", "cmd_pos_xy"), plot_com_tracking_map),
        (
            "ankle torque plot",
            ("t", "adaptive_ankle_torque", "applied_ankle_torque"),
            plot_ankle_torque,
        ),
    )

    for label, required, plot_fn in plots:
        if all(has_dat(name) for name in required):
            plot_fn()
        else:
            missing = ", ".join(f"{name}.dat" for name in required if not has_dat(name))
            print(f"Skipping {label}; missing {missing}. Run deploy_mujoco.py first.")


if __name__ == "__main__":
    main()
