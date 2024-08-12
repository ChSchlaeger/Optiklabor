import numpy as np


def rotation_phi(r, phi):
    """rotate vector/coordinate system in phi direction"""
    r_phi = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    return np.dot(r_phi, r)


def rotation_theta(r, theta):
    """rotate vector/coordinate system in theta direction"""
    r_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(r_theta, r)


def goniometer_to_angle(alpha, beta, gamma, delta):
    """transform goniometer parameters to outgoing and incident direction"""
    alpha = -alpha
    theta_in = np.arccos(np.cos(alpha) * np.cos(beta))
    theta_out = np.arccos(np.cos(alpha) * np.cos(delta - beta))

    s1 = np.sin(alpha) * np.cos(gamma) * np.cos(beta - delta) + np.sin(gamma) * np.sin(beta - delta)
    s2 = np.sin(alpha) * np.cos(gamma) * np.cos(beta) + np.sin(gamma) * np.sin(beta)

    if abs(np.sqrt(1 - np.cos(alpha) ** 2 * np.cos(beta - delta) ** 2)) < 1e-5:
        phi_out = np.pi
    else:
        phi_out_cos = (np.cos(beta - delta) * np.sin(alpha) * np.sin(gamma) - np.sin(beta - delta) * np.cos(
            gamma)) / \
                      np.sqrt(1 - np.cos(alpha) ** 2 * np.cos(beta - delta) ** 2)
        if phi_out_cos > 1:
            phi_out_cos = 1
        elif phi_out_cos < -1:
            phi_out_cos = -1
        phi_out = np.arccos(phi_out_cos)
        if s1 < 0:  # and s2 != 0:
            phi_out = 2 * np.pi - phi_out

    if abs(np.sqrt(1 - np.cos(alpha) ** 2 * np.cos(beta) ** 2)) < 1e-5:
        phi_in = 0.0
    else:
        phi_in_cos = (-np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.cos(beta) * np.sin(gamma)) / \
                     np.sqrt(1 - np.cos(alpha) ** 2 * np.cos(beta) ** 2)
        if phi_in_cos > 1:
            phi_in_cos = 1
        elif phi_in_cos < -1:
            phi_in_cos = -1
        phi_in = np.arccos(phi_in_cos)
        if s2 < 0:  # and s1 != 0:
            phi_in = 2 * np.pi - phi_in

    return theta_out, phi_out, theta_in, phi_in
