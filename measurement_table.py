import os
from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geometry import *

# TODO:
#  - output should be written in the write_table function
#  - create a class for the measurement table
#  - what is N_to, N_pp, N_tp, N_po? What is alpha, beta, gamma, delta?
#  - what is index and index2?
#  - add a function to estimate the measurement time for a given table
#  - try to optimize the measurement time
#  - add a gooey implementation to create the measurement table

"""
Create a table to measure the BRDF of a material. The outgoing direction is constant - the parametrisation
direction is varying.
Only reflection is measured -> delta is in [0...180°[
Input:
    -type of parametrisation (half-angle or incident angle)
    -outgoing angles theta_o and phi_o (for isotropic set phi_o = 0 -> surface tangent points to omega_o direction)
    -camera or spectrometer?
    -spotsize (1 ... 6)
    -divergence (1 ... 5)
    -number of samples for each angle
        - N_to
        - N_pp
        - N_tp
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


class MeasurementTable:
    def __init__(self, parameterization, cam_or_spectro, spotsize, divergence,
                 N_tp, N_pp, N_to, N_po, max_grad,
                 change_gamma, detector_spotsize, light_source_spotsize):
        self.parameterization = parameterization
        self.spotsize = spotsize
        self.divergence = divergence

        if cam_or_spectro:
            self.spectrometer = "Yes"
            self.camera = "No"
        else:
            self.spectrometer = "No"
            self.camera = "Yes"


        self.N_tp = N_tp
        self.N_pp = N_pp
        self.N_to = N_to
        self.N_po = N_po
        self.max_grad = max_grad
        self.change_gamma = change_gamma
        self.detector_spotsize = detector_spotsize
        self.light_source_spotsize = light_source_spotsize

        self.output_list = []
        self.angle_list = []
        self.output_df = None
        self.index = 0
        self.index2 = 0

    @staticmethod
    def test_angles(th_o, ph_o, th_i, ph_i, a, b, c, d):

        a, b, c, d = [x * np.pi / 180 for x in [a, b, c, d]]
        th_o_2, ph_o_2, th_i_2, ph_i_2 = goniometer_to_angle(a, b, c, d)
        angles = np.array([th_o_2, ph_o_2, th_i_2, ph_i_2]) * 180 / np.pi

        diff = (ph_i_2 - ph_o_2) * 180 / np.pi
        if diff > 180:
            diff = diff - 360
        elif diff < -180:
            diff = diff + 360

        if light_source_spotsize / np.cos(b) > detector_spotsize / np.cos(d - b):
            index_2 = True
        else:
            index_2 = False

        if th_o < 1e-5:
            cond3 = (abs(ph_i_2 * 180 / np.pi) - abs(ph_i * 180 / np.pi)) % 180 < 1e-5 or (
                    abs(ph_i_2 * 180 / np.pi) - abs(ph_i * 180 / np.pi)) % 180 - 180 < 1e-5
        else:
            cond3 = abs((abs(diff) - abs(ph_i * 180 / np.pi))) < 1e-5
        if not cond3:
            index_1 = True
        else:
            index_1 = False

        return angles, index_1, index_2

    def find_goniometer_angles(self, theta_out, phi_out, theta_p, phi_p, rounding=1):
        if self.parameterization == 1:
            theta_in, phi_in = find_incident_coord(theta_out, phi_out, theta_p, phi_p)
        else:
            theta_in, phi_in = theta_p, phi_p
        in_vector = angle_to_cartesian(theta_in, phi_in)
        if in_vector[2] >= 0 and np.arccos(in_vector[2]) < max_grad / 180 * np.pi and theta_out < max_grad / 180 * np.pi:
            alpha, beta, gamma, delta = angle_to_goniometer(theta_out, phi_out, theta_in, phi_in)
            if abs(delta * 180 / np.pi) > 8:
                alpha = round(alpha * 180 / np.pi, rounding)
                beta = round(beta * 180 / np.pi, rounding)
                gamma = round(gamma * 180 / np.pi, rounding)
                if change_gamma:
                    if gamma < -90:
                        gamma = gamma + 360
                delta = round(delta * 180 / np.pi, rounding)

                # this is one line in the csv
                output = [beta, alpha, gamma, delta, 0, self.spectrometer, self.camera, self.spotsize, self.divergence]

                # test if the angles are correct
                angles, index_1, index_2 = self.test_angles(theta_out, phi_out, theta_in, phi_in, alpha, beta, gamma, delta)

                return output, angles, index_1, index_2
            else:
                return False, False, False, False
        else:
            return False, False, False, False

    def write_table(self):
        """write table for goniometer measurements.
        to make the measurement faster, sort the table by delta (4th column)"""

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
                        # direct reflection
                        phi_p = 0
                        output, angles, index_1, index_2 = self.find_goniometer_angles(theta_out, phi_out, theta_p, phi_p)
                        if output is not False:
                            self.output_list.append(output)
                            self.angle_list.append(angles)
                        if index_1:
                            self.index += 1
                        if index_2:
                            self.index2 += 1
                    else:
                        # not direct reflexion
                        for i in range(N_pp):
                            # for i in range(int(N_pp/2+1)): # if N_phi = 10: loop over 6 because of reciprocity/isotropy
                            phi_p = 2 * i * np.pi / N_pp
                            output, angles, index_1, index_2 = self.find_goniometer_angles(theta_out, phi_out, theta_p, phi_p)
                            if output is not False:
                                self.output_list.append(output)
                                self.angle_list.append(angles)
                            if index_1:
                                self.index += 1
                            if index_2:
                                self.index2 += 1
                else:
                    for l in range(N_po):
                        phi_out = l * 2 * np.pi / N_po
                        if theta_p == 0:
                            # direct reflection
                            phi_p = 0
                            output, angles, index_1, index_2 = self.find_goniometer_angles(theta_out, phi_out, theta_p, phi_p)
                            if output is not False:
                                self.output_list.append(output)
                                self.angle_list.append(angles)
                            if index_1:
                                self.index += 1
                            if index_2:
                                self.index2 += 1
                        else:
                            # not direct reflection
                            for i in range(N_pp):
                                # for i in range(int(N_pp/2+1)): # if N_phi = 10: loop over 6 because of reciprocity/isotropy
                                phi_p = 2 * i * np.pi / N_pp
                                output, angles, index_1, index_2 = self.find_goniometer_angles(theta_out, phi_out, theta_p, phi_p)
                                if output is not False:
                                    self.output_list.append(output)
                                    self.angle_list.append(angles)
                                if index_1:
                                    self.index += 1
                                if index_2:
                                    self.index2 += 1

        # write output_list into a DataFrame
        self.output_df = pd.DataFrame(self.output_list)
        self.output_df = self.output_df.sort_values(by=[3], ascending=False)

        print("Anzahl Messpunkte:", len(self.output_df))
        print("Anzahl fehlerhafte Messpunkte:", self.index2)

    def save_to_csv(self, save_name: Union[str, None] = None):

        # create the folder if it does not exist
        if not os.path.exists("measurement_procedures"):
            os.makedirs("measurement_procedures")

        # assign a file name if none is given via the method input
        if save_name is None:
            save_name = ("BRDF_measurement_total_"
                         + str(self.N_tp) + str(self.N_pp) + str(self.N_to)
                         + str(self.N_po) + "param" + str(self.parameterization))
            if self.change_gamma:
                save_name += "_gamma-plus-360"

        # save the measurement table to a csv file
        self.output_df.to_csv(
            f"measurement_procedures\\{save_name}.csv", sep=";", decimal=",",
            index=False, header=False
        )

    def create_plots(self):

        # check if angle_list is filled
        if not self.angle_list:
            print("No angles found.")
            return

        # plot the measured points on a sphere
        th_o_plot = 45
        indices = [index for index, values in enumerate(self.angle_list) if abs(values[0] - th_o_plot) < 2]
        remaining_angles = np.array(self.angle_list)[indices]
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
        ax.scatter(x, y, z, color="k", s=20)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_aspect("auto")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    parameterization = 1  # 0: omega_i, 1: omega_h
    cam_or_spectro = 1  # 0: camera, 1: spectrometer
    spotsize = 1
    divergence = 5
    N_tp, N_pp = 24, 16  # 12, 16 #24, 16 #20,12
    N_to, N_po = 10, 1  # 12, 16#8, 1#10, 1#1, 12#10, 1 #16,1
    max_grad = 75
    change_gamma = True
    detector_spotsize = 20
    light_source_spotsize = 5.5

    measurement_table = MeasurementTable(parameterization, cam_or_spectro,
                                         spotsize, divergence,
                                         N_tp, N_pp, N_to, N_po,
                                         max_grad, change_gamma,
                                         detector_spotsize,
                                         light_source_spotsize)

    measurement_table.write_table()
    measurement_table.save_to_csv("test")
    # measurement_table.create_plots()


