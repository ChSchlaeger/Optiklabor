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


class MeasurementPoint:
    def __init__(self):

        # incident and outgoing angles in degrees
        self.theta_out = None
        self.theta_in = None
        self.phi_out = None
        self.phi_in = None

        # incident and outgoing angles in radians
        self.theta_out_deg = None
        self.theta_in_deg = None
        self.phi_out_deg = None
        self.phi_in_deg = None

        # goniometer angles in degrees
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None

        # goniometer angles in radians
        self.alpha_deg = None
        self.beta_deg = None
        self.gamma_deg = None
        self.delta_deg = None

    @classmethod
    def create_measurement_point(cls, theta_out, phi_out, theta_p, phi_p, halfway_parameterization, rounding=1):

        p = cls()

        p.theta_out = theta_out
        p.phi_out = phi_out

        # if halfway parameterization is used, find the incident angles
        # otherwise, phi_p and theta_p are the incident angles
        if halfway_parameterization:
            p.theta_in, p.phi_in = find_incident_angles(theta_out, phi_out, theta_p, phi_p)
        else:
            p.theta_in, p.phi_in = theta_p, phi_p

        # convert to radians
        p.theta_out_deg = np.rad2deg(p.theta_out)
        p.theta_in_deg = np.rad2deg(p.theta_in)
        p.phi_out_deg = np.rad2deg(p.phi_out)
        p.phi_in_deg = np.rad2deg(p.phi_in)

        # calculate goniometer angles
        p.alpha, p.beta, p.gamma, p.delta = angle_to_goniometer(p.theta_out, p.phi_out,
                                                                p.theta_in, p.phi_in)

        # convert to radians
        p.alpha_deg = round(np.rad2deg(p.alpha), rounding)
        p.beta_deg = round(np.rad2deg(p.beta), rounding)
        p.gamma_deg = round(np.rad2deg(p.gamma), rounding)
        p.delta_deg = round(np.rad2deg(p.delta), rounding)

        return p


class MeasurementTable:
    def __init__(self, halfway_parameterization, cam_or_spectro, spotsize, divergence,
                 N_tp, N_pp, N_to, N_po, max_angle,
                 change_gamma, detector_spotsize, light_source_spotsize):
        self.halfway_parameterization = halfway_parameterization
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
        self.max_angle = max_angle
        self.change_gamma = change_gamma
        self.detector_spotsize = detector_spotsize
        self.light_source_spotsize = light_source_spotsize

        self.output_list = []
        self.angle_list = []
        self.output_df = None

    def find_goniometer_angles(self, theta_out, phi_out, theta_p, phi_p):

        # create a measurement point object to store all the angles
        p = MeasurementPoint.create_measurement_point(
            theta_out, phi_out, theta_p, phi_p, self.halfway_parameterization
        )

        incident_vector = angle_to_cartesian(p.theta_in, p.phi_in)
        if incident_vector[2] >= 0 and np.arccos(incident_vector[2]) < np.deg2rad(self.max_angle) and p.theta_out < np.deg2rad(self.max_angle):

            if abs(p.delta_deg) > 8:
                if self.change_gamma and p.gamma_deg < -90:
                    p.gamma_deg = p.gamma_deg + 360

                # check if detector spotsize is large enough for the incident beam
                if self.light_source_spotsize / np.cos(p.beta) > self.detector_spotsize / np.cos(p.delta - p.beta):
                    raise ValueError("Detector spotsize is too small for incident beam.")

                # append to the lists
                output = [p.beta_deg, p.alpha_deg, p.gamma_deg, p.delta_deg, 0, self.spectrometer,
                          self.camera, self.spotsize, self.divergence]
                self.angle_list.append([p.alpha, p.beta, p.gamma, p.delta])
                self.output_list.append(output)

    def write_table(self):
        """write table for goniometer measurements.
        to make the measurement faster, sort the table by delta (4th column)"""

        for k in range(self.N_to):
            if self.N_to == 1:
                theta_out = np.pi / 4
            else:
                theta_out = k * np.pi / 2 / self.N_to
            for j in range(self.N_tp):
                if self.N_tp == 1:
                    theta_p = np.pi / 4
                else:
                    theta_p = j * np.pi / 2 / self.N_tp
                if theta_out == 0 or self.N_po == 1:
                    phi_out = np.pi
                    if theta_p == 0:
                        # direct reflection
                        phi_p = 0
                        self.find_goniometer_angles(theta_out, phi_out, theta_p, phi_p)
                    else:
                        # not direct reflexion
                        for i in range(self.N_pp):
                            # for i in range(int(N_pp/2+1)): # if N_phi = 10: loop over 6 because of reciprocity/isotropy
                            phi_p = 2 * i * np.pi / self.N_pp
                            self.find_goniometer_angles(theta_out, phi_out, theta_p, phi_p)
                else:
                    for ll in range(self.N_po):
                        phi_out = ll * 2 * np.pi / self.N_po
                        if theta_p == 0:
                            # direct reflection
                            phi_p = 0
                            self.find_goniometer_angles(theta_out, phi_out, theta_p, phi_p)
                        else:
                            # not direct reflection
                            for i in range(self.N_pp):
                                # for i in range(int(N_pp/2+1)): # if N_phi = 10: loop over 6 because of reciprocity/isotropy
                                phi_p = 2 * i * np.pi / self.N_pp
                                self.find_goniometer_angles(theta_out, phi_out, theta_p, phi_p)

        # write output_list into a DataFrame
        self.output_df = pd.DataFrame(self.output_list)
        self.output_df = self.output_df.sort_values(by=[3], ascending=False)

        print("Anzahl Messpunkte:", len(self.output_df))
        print("Anzahl fehlerhafte Messpunkte:", self.number_of_invalid_points)

    def save_to_csv(self, save_name: Union[str, None] = None):

        # create the folder if it does not exist
        if not os.path.exists("measurement_procedures"):
            os.makedirs("measurement_procedures")

        # assign a file name if none is given via the method input
        if save_name is None:
            save_name = ("BRDF_measurement_total_"
                         + str(self.N_tp) + str(self.N_pp) + str(self.N_to)
                         + str(self.N_po) + "param" + str(self.halfway_parameterization))
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

    measurement_table = MeasurementTable(
        halfway_parameterization=1,  # 0: omega_i, 1: omega_h
        cam_or_spectro=1,            # 0: camera, 1: spectrometer
        spotsize=1,
        divergence=5,
        N_tp=24, N_pp=16,            # 12, 16 #24, 16 #20,12
        N_to=10, N_po=1,             # 12, 16#8, 1#10, 1#1, 12#10, 1 #16,1
        max_angle=75,
        change_gamma=True,
        detector_spotsize=20,
        light_source_spotsize=5.5)

    measurement_table.write_table()
    measurement_table.save_to_csv("test")
    # measurement_table.create_plots()


