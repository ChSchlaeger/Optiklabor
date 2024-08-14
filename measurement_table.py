import os
from typing import Union, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TODO:
#  - measure the necessary parameters to estimate the measurement time in the lab
#  - try to optimize the measurement time


def cartesian_to_angle(r: Tuple[float, float, float]):
    """Convert Cartesian coordinates with r=1 into spherical coordinates."""
    theta = np.arccos(r[2] / np.linalg.norm(r))
    phi = np.arctan2(r[1], r[0])
    return theta, phi


def angle_to_cartesian(theta: float, phi: float):
    """convert spherical coordinates with r=1 into cartesian coordinates"""
    return np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))


class MeasurementPoint:
    """
    Class that represents a measurement point with all its angles. All angles
    are calculated when an object is created. Each angle is stored in radians
    and degrees.
    """

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
        """
        Calculate goniometer parameters for given outgoing and incident angles.
        This is more or less the original implementation from Marie and Jan.

        :returns: Goniometer angles alpha, beta, gamma, delta in radians
        """

        # calculate delta
        cos_d = (np.sin(self.theta_in) * np.sin(self.theta_out)
                 * np.cos(self.phi_out - self.phi_in)
                 + np.cos(self.theta_in) * np.cos(self.theta_out))
        if cos_d > 1:
            cos_d = 1
        elif cos_d < -1:
            cos_d = -1
        delta = np.arccos(cos_d)
        if delta < np.deg2rad(28):
            delta = -delta

        # calculate epsilon
        if np.sin(delta) * np.sin(self.theta_in) == 0:
            epsilon = 0
        else:
            cos_e = (np.cos(self.theta_out) - np.cos(delta) * np.cos(self.theta_in)) / (np.sin(delta) * np.sin(self.theta_in))
            sin_e = (np.cos(self.phi_in) * np.cos(delta) - np.cos(self.phi_out) * np.sin(self.theta_out) * np.sin(self.theta_in) - np.cos(
                self.theta_out) * np.cos(self.theta_in) * np.cos(self.phi_in)) / (np.sin(self.phi_in) * np.sin(delta) * np.sin(self.theta_in))
            # Jan changed + to -
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
    """
    Class to create a measurement table with all measurement points.
    The table is created based on the given parameters and can be saved to a csv file.

    The csv table columns are structured as follows:
        - beta:         in degrees
        - alpha:        in degrees
        - gamma:        in degrees
        - delta:        in degrees
        - direction:    0
        - spectrometer: Yes/No
        - camera:       Yes/No
        - spotsize:     (1 ... 6)
        - divergence:   (1 ... 5)
    """

    def __init__(self,
                 halfway_parameterization: int,
                 N_tp: int, N_pp: int,
                 N_to: int, N_po: int,
                 spotsize: float,
                 divergence: int,
                 detector_spotsize: float,
                 light_source_spotsize: float,
                 camera_or_spectrometer: str = "spectrometer",
                 change_gamma: bool = True):

        self.halfway_parameterization = halfway_parameterization
        """ Defines the type of parametrisation (half-angle or incident angle). 
        A value of 0 corresponds to omega_in (i.e. N_tp and N_pp are theta_in and phi_in).
        A value of 1 corresponds to omega_halfway (i.e. N_tp and N_pp are theta_h and phi_h). """
        self.N_tp = N_tp
        """ Number of samples for theta_in/theta_halfway """
        self.N_pp = N_pp
        """ Number of samples for phi_in/phi_halfway """
        self.N_to = N_to
        """ Number of samples for theta_out """
        self.N_po = N_po
        """ Number of samples for phi_out. For isotropic materials, this should be 1. """

        self.spotsize = spotsize
        """ Spotsize mode of the light source. Can be set in the goniometer 
        software and can be (1 ... 6). A value of 1 corresponds to a spotsize of 20 cm."""
        self.divergence = divergence
        """ Divergence value of the light source. Can be set in the goniometer
        software and can be (1 ... 5). """
        self.detector_spotsize = detector_spotsize
        """ Spotsize of the detector in cm """
        self.light_source_spotsize = light_source_spotsize
        """ Spotsize of the light source in cm """

        if camera_or_spectrometer == "spectrometer":
            self.spectrometer = "Yes"
            """ This is written into the output csv. Describes the method of calibration."""
            self.camera = "No"
            """ This is written into the output csv. Describes the method of calibration."""
        elif camera_or_spectrometer == "camera":
            self.spectrometer = "No"
            self.camera = "Yes"
        else:
            raise ValueError("camera_or_spectro must be 'spectrometer' or 'camera'")

        self.change_gamma = change_gamma
        """ This is somehow used to correct some measurement points. Default is True. """

        # attributes to store the measurement points
        self.output_list = []
        """ List to store the measurement points. Used to create the output Dataframe. """
        self.angle_list = []
        """ List to store the angles of the measurement points. Used for plot creation. """
        self.output_df = None
        """ Output Dataframe that is converted into the csv output """

    def _calculate_measurement_point(self, theta_out: float, phi_out: float,
                                     theta_p: float, phi_p: float, max_angle: Optional[float] = 75.):
        """
        Calculate the measurement point and append it to the output list.
        The method first creates a MeasurementPoint object with all its angles
        and then checks if the point is valid with a few conditions the points
        must fulfill. If the point is valid, it is appended to the output list.

        :param theta_out: Outgoing theta angle in radians.
        :param phi_out: Outgoing phi angle in radians.
        :param theta_p: Incident or halfway theta angle in radians.
        :param phi_p: Incident or halfway phi angle in radians.
        :param max_angle: Maximum angle in degrees for the incident and outgoing angles.
        """

        # create a MeasurementPoint object to calculate and store all the angles
        p = MeasurementPoint(theta_out, phi_out, theta_p, phi_p,
                             self.halfway_parameterization)

        # limit measurement points to the upper hemisphere
        incident_vector = angle_to_cartesian(p.theta_in, p.phi_in)
        if incident_vector[2] < 0:
            return

        # skip the point if the incident angle exceeds the maximum allowed angle
        if max_angle is not None:
            if np.arccos(incident_vector[2]) >= np.deg2rad(max_angle) or p.theta_out >= np.deg2rad(max_angle):
                return

        # skip measurement points in which the detector blocks the light source
        if abs(p.delta_deg) < 8:
            return

        # skip the point if the detector spotsize is not large enough for the incident beam
        if self.light_source_spotsize / np.cos(p.beta) > self.detector_spotsize / np.cos(p.delta - p.beta):
            return

        # change gamma stuff - I don't really understand this yet
        if self.change_gamma and p.gamma_deg < -90:
            p.gamma_deg = p.gamma_deg + 360

        # append the valid measurement point to the lists
        output = [p.beta_deg, p.alpha_deg, p.gamma_deg, p.delta_deg, 0, self.spectrometer,
                  self.camera, self.spotsize, self.divergence]
        self.angle_list.append(np.array([p.theta_out_deg, p.phi_out_deg, p.theta_in_deg, p.phi_in_deg]))
        self.output_list.append(output)

    def generate(self):
        """
        Generates the measurement table for goniometer measurements. The method
        loops over all possible angles and calls the _calculate_measurement_point
        method to create the measurement points. Ultimately, the method creates
        a DataFrame from the output_list and sorts it by the delta angle.
        """

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
                for ll in range(self.N_po):
                    if theta_out == 0 or self.N_po == 1:
                        phi_out = np.pi
                    else:
                        phi_out = ll * 2 * np.pi / self.N_po

                    # iterate N_pp times over phi_p
                    for i in range(self.N_pp):
                        if theta_p == 0:  # direct reflection -> phi_p is 0 per definition
                            phi_p = 0
                        else:
                            phi_p = i * 2 * np.pi / self.N_pp  # not direct reflection
                        self._calculate_measurement_point(theta_out, phi_out, theta_p, phi_p)

        # write output_list into a DataFrame
        self.output_df = pd.DataFrame(self.output_list)
        self.output_df = self.output_df.drop_duplicates()
        self.output_df = self.output_df.sort_values(by=[3], ascending=False)

    def save_to_csv(self, save_name: Union[str, None] = None):
        """
        Save the measurement table to a csv file.

        :param save_name: Filename or path to save the table to.
        """

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

    def create_plots(self, theta_out_plot: float = 45.,
                     phi_out_plot: float = 180.):
        """
        Creates 2D and 3D plots of the measurement points on a sphere based
        on filtered angles.

        :param theta_out_plot: Target theta_out angle for filtering.
        :param phi_out_plot: Target phi_out angle for filtering.
        """

        # check if angle_list is filled
        if not self.angle_list:
            print("No angles found.")
            return

        # filter the measurement points by theta_out and phi_out
        indices_theta_out = [index for index, values in enumerate(self.angle_list) if abs(values[0] - theta_out_plot) < 2]
        remaining_angles = np.array(self.angle_list)[indices_theta_out]
        indices_phi_out = [index for index, values in enumerate(remaining_angles) if abs(values[1] - phi_out_plot) < 2]
        final_angles = remaining_angles[indices_phi_out]

        # get the plot data
        x, y, z = np.array([angle_to_cartesian(np.deg2rad(values[2]), np.deg2rad(values[3])) for values in final_angles]).T

        # Set up subplots: 1 row, 2 columns
        fig = plt.figure(figsize=(14, 6))
        ax2d = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection='3d')

        # 2D plot
        ax2d.scatter(x, y)
        ax2d.axis('equal')
        ax2d.add_patch(plt.Circle((0, 0), 1, alpha=0.2))
        ax2d.set_title("2D Plot")
        ax2d.set_xlabel('X axis')
        ax2d.set_ylabel('Y axis')

        # 3D plot
        ax3d.set_box_aspect([1, 1, 0.5])  # Aspect ratio is 1:1:1 (equal axes)
        ax3d.scatter(x, y, z)
        ax3d.axis('equal')
        u = np.linspace(0, np.pi/2, 100)
        v = np.linspace(0, 2 * np.pi, 100)  # Cover full azimuthal range
        x_sphere = np.outer(np.sin(u), np.cos(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.cos(u), np.ones_like(v))
        ax3d.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.1, rstride=5, cstride=5)
        ax3d.set_title("3D Plot")
        ax3d.set_xlabel('X axis')
        ax3d.set_ylabel('Y axis')
        ax3d.set_zlabel('Z axis')

        plt.show()

    def estimate_measurement_duration(self):
        """
        Estimates and prints the measurement time for the given measurement table.
        The estimation is based on the time per measurement point and the velocity
        to adjust each angle.
        """

        # check if output_df is present
        if self.output_df is None:
            print("No measurement table found.")
            return

        # parameters for the estimation -> need to be measured in the lab
        measurement_time_per_point = 30  # in seconds
        velocity_alpha = 2  # in degrees per second
        velocity_beta = 2   # in degrees per second
        velocity_gamma = 2  # in degrees per second
        velocity_delta = 2  # in degrees per second
        # -> are angles adjusted at the same time or one after the other?
        parallel_adjustments = True

        # starting conditions -> is this correct?
        old_alpha = 0
        old_beta = 0
        old_gamma = 0
        old_delta = 0
        duration = 0

        # iterate over all measurement points and calculate duration
        for index, row in self.output_df.iterrows():
            alpha = row[1]
            beta = row[0]
            gamma = row[2]
            delta = row[3]

            duration += measurement_time_per_point
            if parallel_adjustments:
                duration += max(abs(alpha - old_alpha) / velocity_alpha,
                                abs(beta - old_beta) / velocity_beta,
                                abs(gamma - old_gamma) / velocity_gamma,
                                abs(delta - old_delta) / velocity_delta)
            else:
                duration += (abs(alpha - old_alpha) / velocity_alpha
                             + abs(beta - old_beta) / velocity_beta
                             + abs(gamma - old_gamma) / velocity_gamma
                             + abs(delta - old_delta) / velocity_delta)

        # calculate days, hours and minutes and print them if the unit is reached
        duration = round(duration)
        days = duration // (24 * 3600)
        duration = duration % (24 * 3600)
        hours = duration // 3600
        duration %= 3600
        minutes = round(duration // 60)

        pure_measurement_time = len(self.output_df) * measurement_time_per_point
        print(f"Estimated measurement time: {days} days, {hours} hours, {minutes} minutes.")
        print(f"Share of pure measurement time: {pure_measurement_time/duration:.1f} %.")


if __name__ == "__main__":

    measurement_table = MeasurementTable(
        halfway_parameterization=1,  # 0: omega_i, 1: omega_h
        N_tp=24, N_pp=16,            # 24, 16
        N_to=10, N_po=1,             # 10,  1
        spotsize=1,
        divergence=5,
        detector_spotsize=20,
        light_source_spotsize=5.5)

    measurement_table.generate()
    measurement_table.save_to_csv("test")
    measurement_table.create_plots()
    measurement_table.estimate_measurement_duration()

    print("Number of measurement points:", len(measurement_table.output_df))
