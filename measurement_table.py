import os
from typing import Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TODO:
#  - ein bisschen testen, Plots fixen
#  - add a function to estimate the measurement time for a given table
#  - try to optimize the measurement time
#  - add a gooey implementation to create the measurement table
#  - add docstrings

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


def cartesian_to_angle(r: Tuple[float, float, float]):
    """Convert Cartesian coordinates with r=1 into spherical coordinates."""
    theta = np.arccos(r[2] / np.linalg.norm(r))
    phi = np.arctan2(r[1], r[0])
    return theta, phi


def angle_to_cartesian(theta: float, phi: float):
    """convert spherical coordinates with r=1 into cartesian coordinates"""
    return np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))


class MeasurementPoint:
    def __init__(self, theta_out: float, phi_out: float, theta_p: float,
                 phi_p: float, halfway_parameterization: int,
                 rounding: int = 1):

        # outgoing angles in radians
        self.theta_out = theta_out
        self.phi_out = phi_out

        # incident angles in radians
        if halfway_parameterization:
            # if halfway parameterization is used, calculate the incident angles
            half_vector = angle_to_cartesian(theta_p, phi_p)
            out_vector = angle_to_cartesian(theta_out, phi_out)
            in_vector = 2 * np.dot(out_vector, half_vector) * half_vector - out_vector
            self.theta_in, self.phi_in = cartesian_to_angle(in_vector)
        else:
            # otherwise, phi_p and theta_p are the incident angles
            self.theta_in, self.phi_in = theta_p, phi_p

        # incident and outgoing angles in degrees
        self.theta_out_deg = np.rad2deg(self.theta_out)
        self.theta_in_deg = np.rad2deg(self.theta_in)
        self.phi_out_deg = np.rad2deg(self.phi_out)
        self.phi_in_deg = np.rad2deg(self.phi_in)

        # calculate goniometer angles in radians
        self.alpha, self.beta, self.gamma, self.delta = self._get_goniometer_angles()

        # goniometer angles in degrees
        self.alpha_deg = round(np.rad2deg(self.alpha), rounding)
        self.beta_deg = round(np.rad2deg(self.beta), rounding)
        self.gamma_deg = round(np.rad2deg(self.gamma), rounding)
        self.delta_deg = round(np.rad2deg(self.delta), rounding)

    def _get_goniometer_angles(self):
        """calculate goniometer parameters for given outgoing and incident angles"""

        # calculate delta
        cos_d = (np.sin(self.theta_in) * np.sin(self.theta_out)
                 * np.cos(self.phi_out - self.phi_in)
                 + np.cos(self.theta_in) * np.cos(self.theta_out))
        if cos_d > 1:
            cos_d = 1
        elif cos_d < -1:
            cos_d = -1
        delta = np.arccos(cos_d)
        if delta < 28 * np.pi / 180:
            delta = -delta

        # calculate epsilon
        if np.sin(delta) * np.sin(self.theta_in) == 0:
            epsilon = 0
        else:
            cos_e = (np.cos(self.theta_out) - np.cos(delta) * np.cos(self.theta_in)) / (np.sin(delta) * np.sin(self.theta_in))
            sin_e = (np.cos(self.phi_in) * np.cos(delta) - np.cos(self.phi_out) * np.sin(self.theta_out) * np.sin(self.theta_in) - np.cos(
                self.theta_out) * np.cos(self.theta_in) * np.cos(self.phi_in)) / (np.sin(self.phi_in) * np.sin(delta) * np.sin(self.theta_in))
            # + changed to -
            if cos_e > 1:
                epsilon = 0
            elif cos_e < -1:
                epsilon = np.pi
            else:
                epsilon = np.arccos(cos_e)
                if sin_e < 0:
                    epsilon = 2 * np.pi - epsilon

        # calculate gamma
        if abs(1 - np.sin(epsilon) ** 2 * np.sin(self.theta_in) ** 2) < 1e-7:
            gamma = 0
        else:
            sin_g = ((-np.sin(epsilon) * np.cos(self.theta_in)
                     * np.cos(self.phi_in) + np.cos(epsilon) * np.sin(self.phi_in))
                     / (np.sqrt(1 - np.sin(epsilon) ** 2 * np.sin(self.theta_in) ** 2)))
            cos_g = ((-np.sin(epsilon) * np.cos(self.theta_in) * np.sin(self.phi_in)
                     - np.cos(epsilon) * np.cos(self.phi_in))
                     / (np.sqrt(1 - np.sin(epsilon) ** 2 * np.sin(self.theta_in) ** 2)))  # Jan changed sin to cos
            if cos_g > 1:
                cos_g = 1
            if cos_g < -1:
                cos_g = -1
            if sin_g >= 0:
                gamma = np.arccos(cos_g)
            else:
                gamma = - np.arccos(cos_g)

        # calculate alpha and beta
        alpha = np.arcsin(-np.sin(epsilon) * np.sin(self.theta_in))
        beta = np.arctan(np.cos(epsilon) * np.tan(self.theta_in))

        return -alpha, beta, gamma, delta


class MeasurementTable:
    def __init__(self, halfway_parameterization: int,
                 camera_or_spectrometer: str,
                 spotsize: float, divergence: int,
                 N_tp: int, N_pp: int, N_to: int, N_po: int,
                 change_gamma: bool, detector_spotsize: float,
                 light_source_spotsize: float):
        self.halfway_parameterization = halfway_parameterization
        self.spotsize = spotsize
        self.divergence = divergence

        if camera_or_spectrometer == "spectrometer":
            self.spectrometer = "Yes"
            self.camera = "No"
        elif camera_or_spectrometer == "camera":
            self.spectrometer = "No"
            self.camera = "Yes"
        else:
            raise ValueError("camera_or_spectro must be 'spectrometer' or 'camera'")

        self.N_tp = N_tp
        self.N_pp = N_pp
        self.N_to = N_to
        self.N_po = N_po
        self.change_gamma = change_gamma
        self.detector_spotsize = detector_spotsize
        self.light_source_spotsize = light_source_spotsize

        self.output_list = []
        self.angle_list = []
        self.output_df = None

    def _calculate_goniometer_angles(self, theta_out: float, phi_out: float,
                                     theta_p: float, phi_p: float, max_angle: float = 75.):

        # create a MeasurementPoint object to calculate and store all the angles
        p = MeasurementPoint(theta_out, phi_out, theta_p, phi_p,
                             self.halfway_parameterization)

        incident_vector = angle_to_cartesian(p.theta_in, p.phi_in)
        if incident_vector[2] >= 0 and np.arccos(incident_vector[2]) < np.deg2rad(max_angle) and p.theta_out < np.deg2rad(max_angle):

            if abs(p.delta_deg) > 8:
                if self.change_gamma and p.gamma_deg < -90:
                    p.gamma_deg = p.gamma_deg + 360

                # check if detector spotsize is large enough for the incident beam
                if self.light_source_spotsize / np.cos(p.beta) > self.detector_spotsize / np.cos(p.delta - p.beta):
                    raise ValueError("Detector spotsize is too small for incident beam.")

                # append to the lists
                output = [p.beta_deg, p.alpha_deg, p.gamma_deg, p.delta_deg, 0, self.spectrometer,
                          self.camera, self.spotsize, self.divergence]
                self.angle_list.append([p.alpha_deg, p.beta_deg, p.gamma_deg, p.delta_deg])
                self.output_list.append(output)

    def _iterate_phi_p(self, theta_out: float, phi_out: float, theta_p: float):

        # direct reflection -> phi_p is 0 per definition
        if theta_p == 0:
            phi_p = 0
            self._calculate_goniometer_angles(theta_out, phi_out, theta_p, phi_p)

        # not direct reflection -> iterate N_pp times over phi_p
        else:
            for i in range(self.N_pp):
                phi_p = 2 * i * np.pi / self.N_pp
                self._calculate_goniometer_angles(theta_out, phi_out, theta_p, phi_p)

    def generate(self):
        """write table for goniometer measurements. loop over all possible angles
        and calculate/save the goniometer angles."""

        # iterate N_to times over theta_out
        for k in range(self.N_to):
            if self.N_to == 1:
                theta_out = np.pi / 4
            else:
                theta_out = k * np.pi / 2 / self.N_to

            # iterate N_tp times over theta_p
            for j in range(self.N_tp):
                if self.N_tp == 1:
                    theta_p = np.pi / 4
                else:
                    theta_p = j * np.pi / 2 / self.N_tp

                # iterate N_po times over phi_out
                if theta_out == 0 or self.N_po == 1:
                    phi_out = np.pi
                    self._iterate_phi_p(theta_out, phi_out, theta_p)
                else:
                    for ll in range(self.N_po):
                        phi_out = ll * 2 * np.pi / self.N_po
                        self._iterate_phi_p(theta_out, phi_out, theta_p)

        # write output_list into a DataFrame
        self.output_df = pd.DataFrame(self.output_list)
        self.output_df = self.output_df.sort_values(by=[3], ascending=False)

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

    def create_plots(self, theta_out_plot=45., phi_out_plot=180.):

        # check if angle_list is filled
        if not self.angle_list:
            print("No angles found.")
            return

        # plot the measured points on a sphere
        indices = [index for index, values in enumerate(self.angle_list) if abs(values[0] - theta_out_plot) < 2]
        remaining_angles = np.array(self.angle_list)[indices]
        print(remaining_angles)

        indices_1 = [index for index, values in enumerate(remaining_angles) if abs(values[1] - phi_out_plot) < 2]
        final_angles = remaining_angles[indices_1]
        print(final_angles)

        xyz = np.array([angle_to_cartesian(values[2] * np.pi / 180, values[3] * np.pi / 180) for values in final_angles])
        x = np.array([xyz[i][0] for i in range(len(xyz))])
        y = np.array([xyz[i][1] for i in range(len(xyz))])
        circle = plt.Circle((0, 0), 1, alpha=0.3)
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.axis('equal')
        ax.add_patch(circle)
        plt.show()

        """
        # this has nothing to do with the actual measurement points -> find a better representation
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
        plt.show()"""


if __name__ == "__main__":

    measurement_table = MeasurementTable(
        halfway_parameterization=1,  # 0: omega_i, 1: omega_h
        camera_or_spectrometer="spectrometer",  # either "camera" or "spectrometer"
        spotsize=1,
        divergence=5,
        N_tp=24, N_pp=16,            # 12, 16 #24, 16 #20,12
        N_to=10, N_po=1,             # 12, 16#8, 1#10, 1#1, 12#10, 1 #16,1
        change_gamma=True,
        detector_spotsize=20,
        light_source_spotsize=5.5)

    measurement_table.generate()
    measurement_table.save_to_csv("test")
    measurement_table.create_plots()

    print("Anzahl Messpunkte:", len(measurement_table.output_df))
