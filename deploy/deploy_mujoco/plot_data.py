from pathlib import Path
import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Show the matplotlib window after saving the plot")
    parser.add_argument("--no-save", action="store_true", help="Do not save the PNG plot")
    parser.add_argument(
        "--plot",
        choices=("all", "plate-imu", "com-velocity-tracking"),
        default="all",
        help="Select which plot to generate",
    )
    args = parser.parse_args()

    if args.plot in ("all", "plate-imu"):
        plot_plate_imu(save=not args.no_save, show=args.show)
    if args.plot in ("all", "com-velocity-tracking"):
        required = ("com_vel_plate", "cmd_vel", "com_vel_tracking_error")
        if all(has_dat(name) for name in required):
            plot_com_velocity_tracking_error(save=not args.no_save, show=args.show)
        else:
            missing = ", ".join(f"{name}.dat" for name in required if not has_dat(name))
            print(f"Skipping COM velocity tracking plot; missing {missing}. Run deploy_mujoco.py first.")


if __name__ == "__main__":
    main()
