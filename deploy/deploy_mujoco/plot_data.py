from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Show the matplotlib window after saving the plot")
    parser.add_argument("--no-save", action="store_true", help="Do not save the PNG plot")
    args = parser.parse_args()
    plot_plate_imu(save=not args.no_save, show=args.show)


if __name__ == "__main__":
    main()
