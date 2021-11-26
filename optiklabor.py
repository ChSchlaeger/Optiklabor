from math import acos, cos, degrees, radians, sin, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gooey import Gooey, GooeyParser


def main():
    """The main method that runs the program.
    """
    (spectral_dir, reference_dir, filter_dir, output_dir, save_intermediate,
     correction_factor, plot_angle) = get_command_line_arguments()
    optiklabor = Optiklabor(
        spectral_dir, reference_dir, filter_dir, output_dir, save_intermediate,
        correction_factor, plot_angle
    )
    optiklabor.build()


@Gooey(
    program_name='Optiklabor',
    navigation='TABBED',
    default_size=(700, 700)
)
def get_command_line_arguments():
    """Gets the command line arguments from the Gooey

    Returns
    -------
    spectral : pathlib.Path
        The folder containing the spectral data
    reference : pathlib.Path or None
        The path to the reference file. Can be None
    filter : pathlib.Path
        The path to the filter file
    output : pathlib.Path
        The folder where the output files will be stored
    intermediate : bool
        Determines if the intermediate files are stored to the disk
    correction_factor : str
        The correction factor to use
    plot_angle : str
        Determines if the final plot is shown as a function of the entrance or
        exit angle

    """
    parser = GooeyParser()
    subs = parser.add_subparsers(dest='command')
    # tab with seperate folders
    separate = subs.add_parser(
        'separate', prog='Folders',
        help='Choose separate folders for the different files'
    )
    separate.add_argument(
        '-s', '--spectral_data', dest='spectral_dir',
        help='Folder containing the spectral data', widget='DirChooser',
        required=True
    )
    separate.add_argument(
        '-o', '--output', dest='output_dir',
        help='Save the output files to this folder', widget='DirChooser'
    )
    separate.add_argument(
        '-r', '--reference_data', dest='reference_file',
        help='File containing the reference data', widget='FileChooser'
    )
    separate.add_argument(
        '-f', '--filter', dest='filter_file',
        help='File containing the filter', widget='FileChooser',
        required=True
    )
    separate.add_argument(
        '-i', '--intermediate', dest='save_intermediate',
        help='Save the intermediate results', action='store_true'
    )
    separate.add_argument(
        '-c', '--correction_factor', dest='correction_factor',
        metavar='Correction factor', widget='Dropdown',
        choices=['No correction', '1/cos(entrance angle)', '1/cos(exit angle)'],
        default='No correction', required=True
    )
    separate.add_argument(
        '-p', '--plot-angle', dest='plot_angle',
        metavar='Plot as function of angle', widget='Dropdown',
        choices=['exit', 'entrance'],
        default='exit'
    )
    # post tabs
    args = parser.parse_args()
    spectral = Path(args.spectral_dir)
    reference = args.reference_file
    if reference:
        reference = Path(args.reference_file)
    output = args.output_dir
    if output:
        output = Path(output)
    else:
        output = spectral / 'Output'
        output.mkdir(exist_ok=True)
    filter = Path(args.filter_file)
    intermediate = args.save_intermediate
    correction_factor = args.correction_factor
    plot_angle = args.plot_angle
    return (spectral, reference, filter, output, intermediate,
            correction_factor, plot_angle)


def in_ranges(x, b):
    """Groups a series into custom bins. Returns a Dataframe with the index of
    x and one column for each tuple in b. Entries that are entirely within a
    bin are weighted with 1 while entries on the edge are weighted with 0.5.

    Parameters
    ----------
    x : pd.Series
    b : list of tuples in the form [(lb, ub)]

    Returns
    -------
    pd.DataFrame

    """
    return [1 if ((x > y[0]) & (x < y[1])) else 0.5
            if ((x >= y[0]) & (x <= y[1])) else 0 for y in b]


class Optiklabor:
    def __init__(self, spectral_dir, reference_dir, filter_dir, output_dir,
                 save_intermediate, correction_factor, plot_angle):
        """Initializes an Optiklabor instance

        Parameters
        ----------
        spectral_dir : pathlib.Path
            The folder containing the spectral data
        reference_dir : pathlib.Path or None
            The path to the reference file. Can be None
        filter_dir : pathlib.Path
            The path to the filter file
        output_dir : pathlib.Path
            The folder where the output files will be stored
        save_intermediate : bool
            Determines if the intermediate files are stored to the disk
        correction_factor : str
            The correction factor to use
        plot_angle : str
            Determines if the final plot is shown as a function of the entrance or
            exit angle
        """
        self.spectral_dir = spectral_dir
        self.reference_dir = reference_dir
        self.filter_dir = filter_dir
        self.output_dir = output_dir
        self.save_intermediate = save_intermediate
        self.correction_factor = correction_factor
        self.plot_angle = plot_angle

        self.ref_data = None
        self.filter_data = None
        self.output_data = None

    def build(self):
        """Fills the Optiklabor with data by calling separate methods that
        gather sub-data themselves.

        Returns
        -------

        """
        self.read_reference_data()
        spectral_files = self.spectral_dir.glob('*.txt')
        output_data = pd.DataFrame()
        for spectral_file in spectral_files:
            intermediate_data, angle_data, meta_data = IntermediateData.build(
                spectral_file, self.ref_data, self.correction_factor,
                self.output_dir, self.save_intermediate
            )
            new_output = OutputData.build(
                intermediate_data, angle_data, meta_data, self.filter_dir)
            output_data = pd.concat([output_data, new_output], axis=1)
        self.output_data = output_data.sort_index(axis=1)
        self.sum_output_data()
        self.write_output_data()
        self.plot_output_data()

    def read_reference_data(self):
        """Reads the reference data if a reference directory is given. Returns
        an empty DataFrame if not

        Returns
        -------

        """
        if self.reference_dir:
            self.ref_data = pd.read_csv(
                self.reference_dir, sep='\t', skiprows=1, header=0, index_col=0,
                decimal=',', engine='python'
            )
            self.ref_data.index = self.ref_data.index.drop_duplicates()
        else:
            self.ref_data = pd.DataFrame()

    def sum_output_data(self):
        """Sums the output data over all wavelengths.

        Returns
        -------

        """
        self.output_data = self.output_data.sum().unstack('Parameter')

    def write_output_data(self):
        """Writes the output data to a csv file. Converts angles from radians
        to degrees.

        Returns
        -------

        """
        # convert radians to degrees
        output_data = self.output_data.copy()
        output_data = output_data.reset_index()
        cols = ['AzimuthEntranceAngle', 'EntranceAngle', 'AzimuthExitAngle',
                'ExitAngle']
        for col in cols:
            output_data[col] = output_data[col].apply(degrees)
        output_file = self.output_dir / 'output.txt'
        with open(output_file, 'w') as f:
            f.write(f'#CorrectionFactor={self.correction_factor}\n')
            output_data.to_csv(f, sep='\t', index=False)

    def plot_output_data(self):
        """Plots the output data.

        Returns
        -------

        """
        if self.plot_angle == 'exit':
            angle = 'ExitAngle'
            azimuth = 'AzimuthExitAngle'
        elif self.plot_angle == 'entrance':
            angle = 'EntranceAngle'
            azimuth = 'AzimuthEntranceAngle'
        else:
            raise ValueError(f'Direction {self.plot_angle} is not supported')
        param = ' Reflexionsgrad [-]'
        cols = [angle, azimuth, param]
        reflection = self.output_data.reset_index()[cols]
        # convert to cartesian coordinates
        theta = reflection[angle]
        phi = reflection[azimuth]
        rho = reflection[param]
        rho[rho < 0] = 0
        # from spherical to cartesian coordinates
        x = rho * np.cos(phi) * np.sin(theta)
        y = rho * np.sin(phi) * np.sin(theta)
        z = rho * np.cos(theta)
        xlabel = 'x'
        ylabel = 'y'
        zlabel = 'Reflection'
        strfile = f'polar_{self.plot_angle}.png'
        self.plot_3d(x, y, z, xlabel, ylabel, zlabel, strfile)

    def plot_3d(self, x, y, z, xlabel, ylabel, zlabel, strfile):
        """Plots a 3D surface.

        Parameters
        ----------
        x : pd.Series or list
            The x data
        y : pd.Series or list
             The y data
        z : pd.Series or list
            The z data
        xlabel : str
            The x label
        ylabel : str
            The y label
        zlabel : str
            The z label
        strfile : str
            The filename

        Returns
        -------

        """
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.view_init(elev=30, azim=-30)
        plt.colorbar(im, location='left', shrink=0.6, pad=0.03)
        plt.show()
        plt.savefig(self.output_dir / strfile, dpi=300)


class IntermediateData:
    def __init__(self, correction_factor):
        """Initializes an object that generates the intermediate data

        Parameters
        ----------
        correction_factor : str
            The correction factor to use

        """
        self.correction_factor = correction_factor

        self.intermediate_data = None

        self.correction_angle = None
        self.ang_entrance = None
        self.ang_exit = None
        self.azi_entrance = None
        self.azi_exit = None
        self.angle_data = dict()

    @classmethod
    def build(cls, spectral_file, ref_data, correction_factor, output_dir,
              save_intermediate):
        """Classmethod that builds an IntermediateData object from spectral data
        and returns the relevant data

        Parameters
        ----------
        spectral_file : pathlib.Path
            The file containing the spectral data
        ref_data : pd.DataFrame
            A DataFrame containing the reference data. Can be empty
        correction_factor : str
            The correction factor to use
        output_dir : pathlib.Path
            The folder where the output files will be stored
        save_intermediate : bool
            Determines if the intermediate files are stored to the disk

        Returns
        -------
        intermediate_data : pd.Series
            The intermediate data
        angle_data : dict
            The exit and entrance angles and their azimuths
        meta_data : dict
            The meta data of the measurements given by the spectral file

        """
        intermediate = cls(correction_factor)
        spectral_data, meta_data = intermediate.read_spectral_data(
            spectral_file)
        intermediate.get_angles(meta_data)
        intermediate.update_angle_data()
        intermediate_data = intermediate.get_data(
            spectral_data, ref_data, meta_data,
            output_dir / f'intermediate_{spectral_file.stem}.txt',
            save_intermediate
        )
        return intermediate_data, intermediate.angle_data, meta_data

    def read_spectral_data(self, spectral_file):
        """Reads the spectral data

        Parameters
        ----------
        spectral_file : pathlib.Path
            The file containing the spectral data

        Returns
        -------
        spectral_data : pd.Series
            The spectral data
        meta_data : dict
            The meta data of the measurements given by the spectral file

        """
        # get actual data
        spectral_data = pd.read_csv(
            spectral_file, sep=r"\t", skiprows=10, header=None, index_col=0,
            engine='python'
        )
        spectral_data = pd.Series(spectral_data.iloc[:, 0])
        spectral_data.name = spectral_file.stem
        spectral_data.index.name = 'Wellenlaenge'
        # get meta data
        meta_data = dict()
        with open(spectral_file, 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for i, line in enumerate(f) if i < 10]
        for line in lines:
            key, value = line.split('=', 1)
            key = key.replace('#', '')
            meta_data[key] = value
        return spectral_data, meta_data

    def get_angles(self, meta_data):
        """Calculates the entrance and exit angle and their azimuths from the
        meta data given by the spectral data

        Parameters
        ----------
        meta_data : dict
            The meta data of the measurements given by the spectral file

        Returns
        -------

        """
        if self.correction_factor == 'No correction':
            self.correction_angle = 1
        alpha = radians(float(meta_data['SampleTilt']) * -1)
        beta = radians(float(meta_data['SampleAngle']))
        gamma = radians(float(meta_data['SampleRotation']))
        delta = radians(float(meta_data['DetectorAngle']))
        # entrance and exit angle
        self.ang_entrance = acos(cos(alpha) * cos(beta))
        self.ang_exit = acos(cos(alpha) * cos(delta - beta))
        if self.correction_factor == '1/cos(entrance angle)':
            self.correction_angle = self.ang_entrance
        elif self.correction_factor == '1/cos(exit angle)':
            self.correction_angle = self.ang_exit
        if self.correction_angle < 1e-12:
            self.correction_angle = 1
            self.correction_factor = 'No correction'
        # azimuths
        cos_azi_exit = ((cos(beta - delta) * sin(alpha) * sin(gamma)
                         - sin(beta - delta) * cos(gamma))
                        / sqrt(1 - cos(alpha) ** 2 * cos(beta - delta) ** 2))
        self.azi_exit = acos(cos_azi_exit)
        condition_exit = (sin(gamma) * sin(beta - delta)
                          + sin(alpha) * cos(gamma) * cos(beta - delta))
        if condition_exit < 0:
            self.azi_exit = radians(360) - self.azi_exit
        cos_azi_entrance = ((-sin(beta) * cos(gamma) + sin(alpha) * cos(beta)
                             * sin(gamma))
                            / sqrt(1 - cos(alpha) ** 2 * cos(beta) ** 2))
        self.azi_entrance = acos(cos_azi_entrance)
        condition_entrance = (sin(gamma) * sin(beta)
                              + sin(alpha) * cos(gamma) * cos(beta))
        if condition_entrance < 0:
            self.azi_entrance = radians(360) - self.azi_entrance

    def get_data(self, spectral_data, ref_data, meta_data, output_file,
                 save_intermediate):
        """Generates the intermediate data

        Parameters
        ----------
        spectral_data : pd.Series
            The spectral data
        ref_data : pd.DataFrame
            A DataFrame containing the reference data. Can be empty
        meta_data : dict
            The meta data of the measurements given by the spectral file
        output_file : pathlib.Path
            The filename for the intermediate data.
            Only used if save_intermediate is True
        save_intermediate : bool
            Specifies if the intermediate data is saved

        Returns
        -------
        intermediate_data : pd.Series or pd.DataFrame
            The intermediate data

        """
        if ref_data.empty:
            intermediate_data = spectral_data
        else:
            new_ind = range(ref_data.index.min(),
                            ref_data.index.max() + 1)
            ref_data = ref_data.reindex(new_ind).interpolate(
                method='linear')
            ref_data = ref_data.reindex(spectral_data.index)
            intermediate_data = ref_data.mul(spectral_data, axis=0)
        intermediate_data /= self.correction_angle
        if save_intermediate:
            with open(output_file, 'w') as f:
                # write angle data
                for key, value in self.angle_data.items():
                    # floats are radians, convert to degrees in meta data
                    if isinstance(value, float):
                        value = round(degrees(value), 2)
                    f.write(f'#{key}={value}\n')
                for key, value in meta_data.items():
                    f.write(f'#{key}={value}\n')
                intermediate_data.to_csv(f, sep='\t')
        return intermediate_data

    def update_angle_data(self):
        """Creates a dictionary containing all angles

        Returns
        -------

        """
        attrs = dict(
            ang_entrance='EntranceAngle',
            ang_exit='ExitAngle',
            azi_entrance='AzimuthEntranceAngle',
            azi_exit='AzimuthExitAngle',
            correction_factor='CorrectionFactor'
        )
        for var, name in attrs.items():
            self.angle_data[name] = getattr(self, var)


class OutputData:
    def __init__(self):
        """Initializes an OutputData instance

        """
        self.filter_data = None

    @classmethod
    def build(cls, intermediate_data, angle_data, meta_data, filter_dir):
        """Classmethod that builds an OutputData instance from intermediate data

        Parameters
        ----------
        intermediate_data : pd.Series or pd.DataFrame
            The intermediate data
        angle_data : dict
            Dictionary containing the angles of the intermediate data
        meta_data : dict
            Dictionary containing the meta data of the original spectral data
        filter_dir : pathlib.Path
            Path to the filter file

        Returns
        -------
        data : pd.DataFrame
            The output data

        """
        output_data = cls()
        output_data.read_filter_data(filter_dir)
        data = output_data.get_data(intermediate_data)
        # means that a reference file was chosen
        if isinstance(data, pd.DataFrame):
            data.columns = pd.MultiIndex.from_product(
                [
                    [angle_data['AzimuthEntranceAngle']],
                    [angle_data['EntranceAngle']],
                    [angle_data['AzimuthExitAngle']],
                    [angle_data['ExitAngle']],
                    data.columns
                ] + [[value] for key, value in meta_data.items()],
                names=['AzimuthEntranceAngle', 'EntranceAngle',
                       'AzimuthExitAngle', 'ExitAngle',
                       'Parameter']
                      + [key for key, value in meta_data.items()]
            )
        # means that no reference file was chosen
        else:
            data = pd.DataFrame(data)
            data.columns = pd.MultiIndex.from_product(
                [
                    [angle_data['AzimuthEntranceAngle']],
                    [angle_data['EntranceAngle']],
                    [angle_data['AzimuthExitAngle']],
                    [angle_data['ExitAngle']],
                    [' Reflexionsgrad [-]']
                ] + [[value] for key, value in meta_data.items()],
                names=['AzimuthEntranceAngle', 'EntranceAngle',
                       'AzimuthExitAngle', 'ExitAngle',
                       'Parameter']
                      + [key for key, value in meta_data.items()]
            )
        return data

    def read_filter_data(self, filter_dir):
        """Reads the filter data

        Parameters
        ----------
        filter_dir : pathlib.Path
            Path to the filter file

        Returns
        -------

        """
        self.filter_data = pd.read_csv(
            filter_dir, sep='\t', header=None, index_col=0, decimal=',')
        self.filter_data.columns = ['weight', 'lower', 'upper']

    def get_data(self, intermediate_data):
        """Gets the output data by filtering the intermediata data with the
        filter data

        Parameters
        ----------
        intermediate_data : pd.Series or pd.DataFrame
            The intermediate data

        Returns
        -------
        output_data : pd.Series
            The output data

        """
        bins = self.filter_data[['lower', 'upper']].to_numpy()
        bins = [tuple(b) for b in bins]
        weights = intermediate_data.reset_index()['Wellenlaenge'].apply(
            lambda x: pd.Series(in_ranges(x, bins), bins))
        weights.index = intermediate_data.index
        weights.columns = self.filter_data.index
        weights = weights @ self.filter_data['weight']
        weights = weights / weights.sum()
        output_data = intermediate_data.mul(weights, axis=0)
        return output_data


if __name__ == '__main__':
    main()
