import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import time

import cv2
import matplotlib.pyplot as plt
import scipy.interpolate


def PolarisationImage_ls(images, angles, mask):
    I = images.reshape((images.shape[0] * images.shape[1], 4))
    A = np.zeros((4, 3))
    A[:, 0] = 1
    A[:, 1] = np.cos(2 * angles)
    A[:, 2] = np.sin(2 * angles)
    # x = np.linalg.pinv(A) @ images.T
    x = np.linalg.lstsq(A, I.T, rcond=None)
    x = x[0].T
    Imax = x[:, 0] + np.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2)
    Imin = x[:, 0] - np.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2)
    Iun = (Imax + Imin) / 2
    # rho = np.divide(Imax - Imin, Imax + Imin)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.true_divide(Imax - Imin, Imax + Imin)
        rho[rho == np.inf] = 0
        rho = np.nan_to_num(rho)
    phi = 0.5 * np.arctan2(x[:, 2], x[:, 1])
    Iun = np.reshape(Iun, (images.shape[0], images.shape[1]))
    rho = np.reshape(rho, (images.shape[0], images.shape[1]))
    phi = np.reshape(phi, (images.shape[0], images.shape[1]))
    Iun2 = np.zeros((images.shape[0], images.shape[1]))
    rho2 = np.zeros((images.shape[0], images.shape[1]))
    phi2 = np.zeros((images.shape[0], images.shape[1]))
    Iun2[mask] = Iun[mask]
    rho2[mask] = rho[mask]
    phi2[mask] = phi[mask]
    return rho2, phi2, Iun2, rho, phi


def rho_diffuse_ls(rho, n):
    theta_d = np.linspace(0, np.pi / 2, 1000)
    rho_d = ((n - 1 / n) ** 2 * np.sin(theta_d) ** 2) / (
        2
        + 2 * n ** 2
        - (n + 1 / n) ** 2 * np.sin(theta_d) ** 2
        + 4 * np.cos(theta_d) * np.sqrt(n ** 2 - np.sin(theta_d) ** 2)
    )
    theta = scipy.interpolate.interp1d(rho_d, theta_d, fill_value="extrapolate")(rho)
    return theta


def rho_spec_ls(rho, n):
    theta_s = np.linspace(0, np.pi / 2, 1000)
    rho_s = (
        2
        * np.sin(theta_s) ** 2
        * np.cos(theta_s)
        * np.sqrt(n ** 2 - np.sin(theta_s) ** 2)
    ) / (
        n ** 2
        - np.sin(theta_s) ** 2
        - n ** 2 * np.sin(theta_s) ** 2
        + 2 * np.sin(theta_s) ** 4
    )
    imax = np.argmax(rho_s)
    rho_s1 = rho_s[:imax]
    theta_s1 = theta_s[:imax]
    theta1 = scipy.interpolate.interp1d(rho_s1, theta_s1, fill_value="extrapolate")(rho)

    rho_s2 = rho_s[imax:]
    theta_s2 = theta_s[imax:]
    theta2 = scipy.interpolate.interp1d(rho_s2, theta_s2, fill_value="extrapolate")(rho)
    return theta1, theta2


def calc_normals_ls(phi, theta, mask):
    N1 = np.cos(phi) * np.sin(theta)
    N2 = np.sin(phi) * np.sin(theta)
    N3 = np.cos(theta)
    N = np.zeros((phi.shape[0], phi.shape[1], 3))
    N[:, :, 0][mask] = N1[mask]
    N[:, :, 1][mask] = N2[mask]
    N[:, :, 2][mask] = N3[mask]
    return N


def main(main_path):
    pose_paths = (main_path / "masks").glob("*.png")
    indices = []
    for el in pose_paths:
        indices.append(int(el.stem))

    start_time = time.time()

    for image_no in indices:
        im0 = cv2.imread(
            (main_path / "images" / f"{image_no}_0.png").as_posix(),
            cv2.IMREAD_GRAYSCALE,
        )
        im45 = cv2.imread(
            (main_path / "images" / f"{image_no}_45.png").as_posix(),
            cv2.IMREAD_GRAYSCALE,
        )
        im90 = cv2.imread(
            (main_path / "images" / f"{image_no}_90.png").as_posix(),
            cv2.IMREAD_GRAYSCALE,
        )
        im135 = cv2.imread(
            (main_path / "images" / f"{image_no}_135.png").as_posix(),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = cv2.imread(
            (main_path / "masks" / f"{image_no}.png").as_posix(),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = np.array(mask, dtype=bool)
        angles = np.array([0, 45, 90, 135]) * np.pi / 180
        n = 2.75

        images = np.zeros((im0.shape[0], im0.shape[1], 4))
        images[:, :, 0][mask] = im0[mask]
        images[:, :, 1][mask] = im45[mask]
        images[:, :, 2][mask] = im90[mask]
        images[:, :, 3][mask] = im135[mask]

        rho2, phi2, Iun2, rho, phi = PolarisationImage_ls(images, angles, mask)
        theta_diff = rho_diffuse_ls(rho2, n)
        theta_spec1, theta_spec2 = rho_spec_ls(rho2, n)
        N_diff = calc_normals_ls(phi2, theta_diff, mask)
        N_spec1 = calc_normals_ls(phi2 + np.pi / 2, theta_spec1, mask)
        N_spec2 = calc_normals_ls(phi2 + np.pi / 2, theta_spec2, mask)

        cv2.imwrite(
            (main_path / "normals" / f"diffuse{image_no}.png").as_posix(),
            N_diff*255,
        )
        cv2.imwrite(
            (main_path / "normals" / f"specular{image_no}_1.png").as_posix(),
            N_spec1*255,
        )
        cv2.imwrite(
            (main_path / "normals" / f"specular{image_no}_2.png").as_posix(),
            N_spec2*255,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-main_path",
        type=str,
        help="path to the folder with polarization 'images', 'masks', 'normals'",
        default="/home/daoyig/MscThesis/polarizationposeresearch/data/data_sample",
    )
    args = parser.parse_args()
    main_path = Path(args.main_path)
    main(main_path)
