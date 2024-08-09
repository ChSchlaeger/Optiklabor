import numpy as np

def rotation_phi(r, phi):
    """rotate vector/coordinate system in phi direction"""
    r_phi = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    return np.dot(r_phi, r)


def rotation_theta(r, theta):
    """rotate vector/coordinate system in theta direction"""
    r_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(r_theta, r)