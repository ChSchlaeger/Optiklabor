# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:52:41 2023

@author: janhen
"""

# TODO:
#  - fix formatting
#  - small refactoring
#  - save generated measurement tables to measurement_procedues folder
#  - add a function to estimate the measurement time for a given table
#  - add a gooey implementation to create the measurement table
#  - fix the csv format -> does this need fixing? I remember some minor issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py

parameterization = 1  # 0: omega_i, 1: omega_h
cam_or_spectro = 1  # 0: camera, 1: spectrometer
spotsize = 1
divergence = 5
N_tp, N_pp = 24, 16  # 12, 16 #24, 16 #20,12
N_to, N_po = 10, 1  # 12, 16#8, 1#10, 1#1, 12#10, 1 #16,1
rnd = 1  # rounding 1
max_grad = 75

change_gamma = True

detector_spotsize = 20
light_source_spotsize = 5.5

"""
Create a table to measure the BRDF of a material. The outgoing direction is constant - the parameterisation
direction is varying.
Only reflection is measured -> delta is in [0...180°[
Input:
    -type of parameterisation (halfangle or incident angle)
    -outgoing angles theta_o and phi_o (for isotropic set phi_o = 0 -> surface tangent points to omega_o direction)
    -camera or spectrometer?
    -spotsize (1 ... 6)
    -divergence (1 ... 5)
    -number of samples for each angle
        - N_to
        - N_pp
        _ N_tp
Output:
    - angles
        - beta
        - alpha
        - gamma
        - delta
    - direction
    - spectrometer bool
    - camera bool
    - spotsize (1 ... 6)
    - divergence (1 ... 5)
"""


# rotating a vector: first theta then phi direction
# rotating a coordinate system: first phi then theta direction and negative angles


def rotation_phi(r, phi):
    """rotate vector/coordinate system in phi direction"""
    R_phi = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    return np.dot(R_phi, r)


def rotation_theta(r, theta):
    """rotate vector/coordinate system in theta direction"""
    r_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(r_theta, r)


def angle_to_cartesian(theta, phi):
    """convert spherical coordinates with r=1 into cartesian coordinates"""
    return np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))


def cartesian_to_angle(r):
    """convert cartesian coordinates with r=1 into spherical coordinates"""
    x = r[0]
    y = r[1]
    z = r[2]
    theta = np.arccos(z / (np.sqrt(x * x + y * y + z * z)))
    phi = np.arctan2(y, x)
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
        if s1 < 0:  # and s2!=0:
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
        if s2 < 0:  # and s1!=0:
            phi_in = 2 * np.pi - phi_in

    return theta_out, phi_out, theta_in, phi_in


def test_angles(th_o, ph_o, th_i, ph_i, a, b, c, d):
    a = a * np.pi / 180
    b = b * np.pi / 180
    c = c * np.pi / 180
    d = d * np.pi / 180
    th_o_2, ph_o_2, th_i_2, ph_i_2 = goniometer_to_angle(a, b, c, d)
    angles = np.array([th_o_2, ph_o_2, th_i_2, ph_i_2]) * 180 / np.pi
    wanted = np.array([th_o, ph_o, th_i, ph_i]) * 180 / np.pi
    diff_th_o = th_o - th_o_2
    diff_ph_o = ph_o - ph_o_2
    diff_th_i = th_i - th_i_2
    diff_ph_i = ph_i - ph_i_2
    diff_list = np.array([diff_th_o, diff_ph_o, diff_th_i, diff_ph_i]) * 180 / np.pi

    # far_away = any(abs(x) > 1 for x in diff_list)
    # if far_away == True:
    far_away = [x for x in range(len(diff_list)) if abs(diff_list[x]) > 1.1]
    if len(far_away) > 0:
        x = far_away[0]
        if x == 1 or x == 3:
            if wanted[x] > 1 and angles[x] < 359:
                print("angles don't fit.", far_away, " wanted:", wanted, "control:", angles)
        else:
            print("angles don't fit.", far_away, " wanted:", wanted, "control:", angles)

    # cond1 = (abs(th_o*180/np.pi - th_o_2*180/np.pi) < 1e-5)
    # cond2 = (abs(th_i*180/np.pi - th_i_2*180/np.pi) < 1e-5)
    diff = (ph_i_2 - ph_o_2) * 180 / np.pi
    if diff > 180:
        diff = diff - 360
    elif diff < -180:
        diff = diff + 360
    phin = ph_i * 180 / np.pi
    if light_source_spotsize / np.cos(b) > detector_spotsize / np.cos(d - b):
        index_2 = True
    else:
        index_2 = False

    if th_o < 1e-5:
        cond3 = (abs(ph_i_2 * 180 / np.pi) - abs(ph_i * 180 / np.pi)) % 180 < 1e-5 or (
                    abs(ph_i_2 * 180 / np.pi) - abs(ph_i * 180 / np.pi)) % 180 - 180 < 1e-5
    else:
        cond3 = abs((abs(diff) - abs(phin))) < 1e-5
    if not cond3:
        index_1 = True
    else:
        index_1 = False
    if a < -np.pi / 2 or a > np.pi / 2:
        print("alpha")
    if b < -np.pi / 2 or b > np.pi / 2:
        print("beta")
    if change_gamma:
        if c < -np.pi / 2 or c > 3 * np.pi / 2:
            print("gamma")
    elif not change_gamma:
        if c < -np.pi or c > np.pi:
            print("gamma")
    # if c <-np.pi or c > np.pi:
    # print("gamma")
    if abs(d) < 8 * np.pi / 180 or d > 192 * np.pi / 180:
        print("delta", d * 180 / np.pi)
    return angles, index_1, index_2


def find_goniometer_angles(theta_out, phi_out, theta_p, phi_p, param, spectr, cam, sp_size, div):
    if param == 1:
        theta_in, phi_in = find_incident_coord(theta_out, phi_out, theta_p, phi_p)
    else:
        theta_in, phi_in = theta_p, phi_p
    in_vector = angle_to_cartesian(theta_in, phi_in)
    if in_vector[2] >= 0 and np.arccos(in_vector[2]) < max_grad / 180 * np.pi and theta_out < max_grad / 180 * np.pi:
        alpha, beta, gamma, delta = angle_to_goniometer(theta_out, phi_out, theta_in, phi_in)
        if abs(delta * 180 / np.pi) > 8:
            """
            xx.append(in_vector[0])
            yy.append(-in_vector[1]) #added -
            zz.append(in_vector[2])
            """
            ar = round(alpha * 180 / np.pi, rnd)
            br = round(beta * 180 / np.pi, rnd)
            cr = round(gamma * 180 / np.pi, rnd)
            if change_gamma == True:
                if cr < -90:
                    cr = cr + 360
            dr = round(delta * 180 / np.pi, rnd)
            ax = str(ar)
            bx = str(br)
            cx = str(cr)
            dx = str(dr)
            bx, ax, cx, dx = bx.replace(".", ","), ax.replace(".", ","), cx.replace(".", ","), dx.replace(".", ",")
            output = [bx, ax, cx, dx, 0, spectr, cam, sp_size, div]
            angles, index_1, index_2 = test_angles(theta_out, phi_out, theta_in, phi_in, ar, br, cr, dr)
            return output, angles, index_1, index_2
        else:
            return False, False, False, False
    else:
        return False, False, False, False


def write_table():
    """write table for goniometer measurements. to make the measurement faster, sort the table by delta (4th column)"""

    if cam_or_spectro:
        spectrometer = "Yes"
        camera = "No"
    else:
        spectrometer = "No"
        camera = "Yes"

    index = 0
    index2 = 0
    output_list = []
    angle_list = []
    for k in range(N_to):
        if N_to == 1:
            theta_out = np.pi / 4
        else:
            theta_out = k * np.pi / 2 / N_to
        for j in range(N_tp):
            if N_tp == 1:
                theta_p = np.pi / 4
            else:
                theta_p = j * np.pi / 2 / N_tp
            if theta_out == 0 or N_po == 1:
                phi_out = np.pi
                if theta_p == 0:
                    # direct reflexion
                    phi_p = 0
                    output, angles, index_1, index_2 = find_goniometer_angles(theta_out, phi_out, theta_p, phi_p,
                                                                              parameterization, spectrometer, camera,
                                                                              spotsize, divergence)
                    if not output == False:
                        output_list.append(output)
                        angle_list.append(angles)
                    if index_1 == True:
                        index += 1
                    if index_2 == True:
                        index2 += 1
                else:
                    # not direct reflexion
                    for i in range(N_pp):
                        # for i in range(int(N_pp/2+1)): # if N_phi = 10: loop over 6 because of reciprocity/isotropy
                        phi_p = 2 * i * np.pi / N_pp
                        output, angles, index_1, index_2 = find_goniometer_angles(theta_out, phi_out, theta_p, phi_p,
                                                                                  parameterization, spectrometer,
                                                                                  camera, spotsize, divergence)
                        if not output == False:
                            output_list.append(output)
                            angle_list.append(angles)
                        if index_1 == True:
                            index += 1
                        if index_2 == True:
                            index2 += 1
            else:
                for l in range(N_po):
                    phi_out = l * 2 * np.pi / N_po
                    if theta_p == 0:
                        # direct reflexion
                        phi_p = 0
                        output, angles, index_1, index_2 = find_goniometer_angles(theta_out, phi_out, theta_p, phi_p,
                                                                                  parameterization, spectrometer,
                                                                                  camera, spotsize, divergence)
                        if not output == False:
                            output_list.append(output)
                            angle_list.append(angles)
                        if index_1 == True:
                            index += 1
                        if index_2 == True:
                            index2 += 1
                    else:
                        # not direct reflexion
                        for i in range(N_pp):
                            # for i in range(int(N_pp/2+1)): # if N_phi = 10: loop over 6 because of reciprocity/isotropy
                            phi_p = 2 * i * np.pi / N_pp
                            output, angles, index_1, index_2 = find_goniometer_angles(theta_out, phi_out, theta_p,
                                                                                      phi_p, parameterization,
                                                                                      spectrometer, camera, spotsize,
                                                                                      divergence)
                            if not output == False:
                                output_list.append(output)
                                angle_list.append(angles)
                            if index_1 is True:
                                index += 1
                            if index_2 is True:
                                index2 += 1

    df = pd.DataFrame(output_list)
    print(np.array(angle_list))
    print("Anzahl Messpunkte", len(df))
    """
    #if not os.path.exists(Measurement_table):
    #    os.makedirs(Measurement_table)
    if change_gamma == True:
        df.to_csv("Measurement_table\\"+ "BRDF_measurement_total_" + str(N_tp) + str(N_pp) + str(N_to) + str(N_po) + "param" + str(parameterization) + "_gamma-plus-360.csv", sep="\t", index=False, header=False)
    else:
        df.to_csv("Measurement_table\\"+ "BRDF_measurement_total_" + str(N_tp) + str(N_pp) + str(N_to) + str(N_po) + "param" + str(parameterization) + ".csv", sep="\t", index=False, header=False)
    """
    print("Anzahl fehlerhafte Messpunkte", index2)

    # if N_to > 1:
    #    th_o_plot = angle_list[int(len(angle_list)/2)][0]
    # else:
    #    th_o_plot = 45
    th_o_plot = 45
    print(th_o_plot)
    print(list(enumerate(angle_list))[0])
    indices = [index for index, values in enumerate(angle_list) if abs(values[0] - th_o_plot) < 2]
    print(len(indices))
    remaining_angles = np.array(angle_list)[indices]
    ph_o_plot = 180
    indices_1 = [index for index, values in enumerate(remaining_angles) if abs(values[1] - ph_o_plot) < 2]
    final_angles = remaining_angles[indices_1]
    xyz = np.array([angle_to_cartesian(values[2] * np.pi / 180, values[3] * np.pi / 180) for values in final_angles])

    x = np.array([xyz[i][0] for i in range(len(xyz))])
    y = np.array([xyz[i][1] for i in range(len(xyz))])
    circle = plt.Circle((0, 0), 1, alpha=0.3)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.axis('equal')
    ax.add_patch(circle)
    plt.show()
    # plt.savefig("Measurement_table\\" + "angles +" str(N_tp) + str(N_pp) + str(N_to) + str(N_po) + "_outgoing-fest.png"")

    """
    # create a sphere to see which points are measured
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
    ax.scatter(xx, yy, zz, color="k", s=20)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("auto")
    plt.tight_layout()
    plt.show()
    """


if __name__ == "__main__":
    write_table()
