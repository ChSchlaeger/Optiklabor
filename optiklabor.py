from math import acos, cos, degrees, radians, sin, sqrt
from pathlib import Path
from scipy.integrate import trapezoid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gooey import Gooey, GooeyParser


def main():
    (spectral_dir, reference_dir, filter_dir, output_dir, save_intermediate,
     correction_factor) = get_command_line_arguments()
    optiklabor = Optiklabor(
        spectral_dir, reference_dir, filter_dir, output_dir, save_intermediate,
        correction_factor
    )
    optiklabor.build()


# @Gooey(
#     program_name='Optiklabor',
#     navigation='TABBED'
# )
def get_command_line_arguments():
    parser = GooeyParser()
    subs = parser.add_subparsers(dest='command')
    # single tab
    single = subs.add_parser(
        'single', prog='Single Folder',
        help='Only choose a single folder. This requires the folder to have '
             'the expected file structure'
    )
    single.add_argument(
        '-t', '--total', dest='total_dir',
        help='Folder containing the data in sub-folders', widget='DirChooser'
    )
    # separate tab
    separate = subs.add_parser(
        'separate', prog='separate Folders',
        help='Choose separate folders for the different files'
    )
    separate.add_argument(
        '-s', '--spectral_data', dest='spectral_dir',
        help='Folder containing the spectral data', widget='DirChooser'
    )
    separate.add_argument(
        '-o', '--output', dest='output_dir',
        help='Save the output files to this folder', widget='DirChooser'
    )
    separate.add_argument(
        '-r', '--reference_data', dest='reference_dir',
        help='File containing the reference data', widget='FileChooser'
    )
    separate.add_argument(
        '-f', '--filter', dest='filter_dir',
        help='Folder containing the filter files', widget='DirChooser'
    )
    separate.add_argument(
        '-i', '--intermediate', dest='intermediate_dir',
        help='Save the intermediate results', action='store_true'
    )
    separate.add_argument(
        '-c', '--correction_factor', dest='correction_factor',
        metavar='Correction factor', widget='Dropdown',
        choices=['No correction', '1/cos(entrance angle)', '1/cos(exit angle)'],
        default='No correction', required=True
    )
    # post tabs
    args = parser.parse_args()
    if args.command == 'single':#TODO
        top = Path(args.total_dir)
        spectral = top / 'Raw'
        reference = top / 'Calibration' / 'Spectrometers'
        output = top / 'Output'
        output.mkdir(exist_ok=True)
        filter = top # TODO
    elif args.command == 'separate':
        spectral = Path(args.spectral_dir)
        reference = Path(args.reference_dir)
        output = Path(args.output_dir)
        filter = Path(args.filter_dir)
        intermediate = args.intermediate_dir
        correction_factor = args.correction_factor
    return spectral, reference, filter, output, intermediate, correction_factor


def in_ranges(x, b):
    return [1 if ((x > y[0]) & (x < y[1])) else 0.5
            if ((x >= y[0]) & (x <= y[1])) else 0 for y in b]


class Optiklabor:
    def __init__(self, spectral_dir, reference_dir, filter_dir, output_dir,
                 save_intermediate, correction_factor):
        self.spectral_dir = spectral_dir
        self.reference_dir = reference_dir
        self.filter_dir = filter_dir
        self.output_dir = output_dir
        self.save_intermediate = save_intermediate
        self.correction_factor = correction_factor

        self.ref_data = None
        self.filter_data = None
        self.output_data = None

    def build(self):
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
        self.integrate_output_data()
        self.write_output_data()
        self.plot_output_data()

    def read_reference_data(self):
        self.ref_data = pd.read_csv(
            self.reference_dir, sep='\t', skiprows=1, header=0, index_col=0,
            decimal=',', engine='python'
        )
        self.ref_data.index = self.ref_data.index.drop_duplicates()

    def integrate_output_data(self):
        output_data = self.output_data.apply(
            lambda y: trapezoid(y, self.output_data.index))
        self.output_data = output_data.unstack('Parameter')

    def write_output_data(self):
        output_file = self.output_dir / 'output.txt'
        with open(output_file, 'w') as f:
            f.write(f'#CorrectionFactor={self.correction_factor}\n')
            self.output_data.to_csv(f, sep='\t')

    def plot_output_data(self, direction='exit'):
        if direction == 'exit':
            angle = 'ExitAngle'
            azimuth = 'AzimuthExitAngle'
        elif direction == 'entrance':
            angle = 'EntranceAngle'
            azimuth = 'AzimuthEntranceAngle'
        else:
            raise ValueError(f'Direction {direction} is not supported')
        param = ' Reflexionsgrad [-]'
        cols = [angle, azimuth, param]
        reflection = self.output_data.reset_index()[cols]
        # convert to cartesian coordinates
        theta = reflection[angle]
        phi = reflection[azimuth]
        rho = reflection[param]
        rho[rho < 0] = 0
        # cylindrical: Probably wrong, because cylindrical only has one angle
        # x = phi * np.cos(theta)
        # y = phi * np.sin(theta)
        # z = rho
        # spherical
        x = rho * np.cos(phi) * np.sin(theta)
        y = rho * np.sin(phi) * np.sin(theta)
        z = rho * np.cos(theta)
        xlabel = 'x'
        ylabel = 'y'
        zlabel = 'Reflection'
        strfile = f'polar_{direction}.png'
        self.polar_plot(x, y, z, xlabel, ylabel, zlabel, strfile)

    def polar_plot(self, x, y, z, xlabel, ylabel, zlabel, strfile):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        im = ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.view_init(elev=30, azim=-30)
        plt.colorbar(im, location='left', shrink=0.6, pad=0.03)
        plt.savefig(self.output_dir / strfile, dpi=300)
        plt.close()


class IntermediateData:
    def __init__(self, correction_factor):
        self.correction_factor = correction_factor

        self.intermediate_data = None

        self.correction_angle = None
        self.ang_entrance = None
        self.ang_exit = None
        self.azi_entrance = None
        self.azi_exit = None
        self.correction_angle = None
        self.angle_data = dict()

    @classmethod
    def build(cls, spectral_file, ref_data, correction_factor, output_dir,
              save_intermediate):
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
        if self.correction_factor == '1':
            return 1
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
        condition = (sin(gamma) * sin(beta - delta)
                     + sin(alpha) * cos(gamma) * cos(beta - delta))
        cos_azi_entrance = ((-sin(beta) * cos(gamma) + sin(alpha) * cos(beta)
                             * sin(gamma))
                            / sqrt(1 - cos(alpha) ** 2 * cos(beta) ** 2))
        cos_azi_exit = ((cos(beta - delta) * sin(alpha) * sin(gamma)
                         - sin(beta - delta) * cos(gamma))
                        / sqrt(1 - cos(alpha) ** 2 * cos(beta - delta) ** 2))
        self.azi_entrance = acos(cos_azi_entrance)
        self.azi_exit = acos(cos_azi_exit)
        if condition < 0:
            self.azi_entrance = radians(360) - self.azi_entrance
            self.azi_exit = radians(360) - self.azi_exit

    def get_data(self, spectral_data, ref_data, meta_data, output_file,
                 save_intermediate):
        # TODO optional keine Referenzdatei bei relativmessung
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
        self.filter_data = None

    @classmethod
    def build(cls, intermediate_data, angle_data, meta_data, filter_dir):
        output_data = cls()
        output_data.read_filter_data(filter_dir)
        data = output_data.get_data(intermediate_data)
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
        return data

    def read_filter_data(self, filter_dir):
        self.filter_data = pd.read_csv(
            filter_dir, sep='\t', header=None, index_col=0, decimal=',')
        self.filter_data.columns = ['weight', 'lower', 'upper']

    def get_data(self, intermediate_data):
        bins = self.filter_data[['lower', 'upper']].to_numpy()
        bins = [tuple(b) for b in bins]
        weights = intermediate_data.reset_index()['Wellenlaenge'].apply(
            lambda x: pd.Series(in_ranges(x, bins), bins))
        weights.index = intermediate_data.index
        weights.columns = self.filter_data.index
        weights = weights @ self.filter_data['weight']
        output_data = intermediate_data.mul(weights, axis=0)
        return output_data


if __name__ == '__main__':
    main()
