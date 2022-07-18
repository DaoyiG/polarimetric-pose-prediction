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

