import numpy as np


# rotating a vector: first theta then phi direction
# rotating a coordinate system: first phi then theta direction and negative angles


def angle_to_cartesian(theta, phi):
    """convert spherical coordinates with r=1 into cartesian coordinates"""
    return np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))


def cartesian_to_angle(r):
    """Convert Cartesian coordinates with r=1 into spherical coordinates."""
    theta = np.arccos(r[2] / np.linalg.norm(r))
    phi = np.arctan2(r[1], r[0])
    return theta, phi


def find_incident_coord(theta_out, phi_out, theta_half, phi_half):
    """calculate angles of incident direction for given outgoing angles and halfway angles"""
    half_vector = angle_to_cartesian(theta_half, phi_half)
    out_vector = angle_to_cartesian(theta_out, phi_out)
    in_vector = 2 * np.dot(out_vector, half_vector) * half_vector - out_vector
    return cartesian_to_angle(in_vector)


def angle_to_goniometer(theta_out, phi_out, theta_in, phi_in):
    """calculate goniometer parameters for given outgoing and incident angles"""
    cos_d = np.sin(theta_in) * np.sin(theta_out) * np.cos(phi_out - phi_in) + np.cos(theta_in) * np.cos(theta_out)
    if cos_d > 1:
        cos_d = 1
    elif cos_d < -1:
        cos_d = -1
    delta = np.arccos(cos_d)
    if delta < 28 * np.pi / 180:
        delta = -delta
    if np.sin(delta) * np.sin(theta_in) == 0:
        epsilon = 0
    else:
        cos_e = (np.cos(theta_out) - np.cos(delta) * np.cos(theta_in)) / (np.sin(delta) * np.sin(theta_in))
        sin_e = (np.cos(phi_in) * np.cos(delta) - np.cos(phi_out) * np.sin(theta_out) * np.sin(theta_in) - np.cos(
            theta_out) * np.cos(theta_in) * np.cos(phi_in)) / (np.sin(phi_in) * np.sin(delta) * np.sin(theta_in))
        # + changed to -
        if cos_e > 1:
            epsilon = 0
        elif cos_e < -1:
            epsilon = np.pi
        else:
            epsilon = np.arccos(cos_e)
            if sin_e < 0:
                epsilon = 2 * np.pi - epsilon

    # delta < 0: epsilon increases/decreases by pi; beta changes sign
    beta = np.arctan(np.cos(epsilon) * np.tan(theta_in))
    if abs(1 - np.sin(epsilon) ** 2 * np.sin(theta_in) ** 2) < 1e-7:
        gamma = 0
    else:
        sin_g = (-np.sin(epsilon) * np.cos(theta_in) * np.cos(phi_in) + np.cos(epsilon) * np.sin(phi_in)) / (
            np.sqrt(1 - np.sin(epsilon) ** 2 * np.sin(theta_in) ** 2))  # added brackets
        cos_g = (-np.sin(epsilon) * np.cos(theta_in) * np.sin(phi_in) - np.cos(epsilon) * np.cos(phi_in)) / (
            np.sqrt(1 - np.sin(epsilon) ** 2 * np.sin(theta_in) ** 2))  # changed sin to cos
        if cos_g > 1:
            cos_g = 1
        if cos_g < -1:
            cos_g = -1
        if sin_g >= 0:
            gamma = np.arccos(cos_g)
        else:
            gamma = - np.arccos(cos_g)

    # delta < 0: gamma increases/decreases by pi; alpha changes sign
    alpha = np.arcsin(-np.sin(epsilon) * np.sin(theta_in))
    return -alpha, beta, gamma, delta


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
        phi_out_cos = (np.cos(beta - delta) * np.sin(alpha) * np.sin(gamma) - np.sin(beta - delta) * np.cos(gamma)) / \
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