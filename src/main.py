from pathlib import Path
import ffmpeg
import numpy as np
from scipy.optimize._lsq import least_squares
from colour import LUT3D, LUT3x1D, write_LUT, cctf_encoding, cctf_decoding

from extract import get_metadata, sample_frames, save_patches
from settings import find_source_gamma, find_target_gamma, read_yml_settings, extract_file_settings
from correction import fit_curve, load_patch_type, remove_outlier_patches, resample_patches, solve_colour_matrix, remove_saturated_patches, sort_patches
from plot import plot_LUT


def load_settings():
    settings_path = Path('settings.yml')
    settings = read_yml_settings(settings_path)

    LUT_settings = settings['LUT']
    files = extract_file_settings(settings)

    return (LUT_settings, files)


def extract_patches(files):
    # TODO: Check existing CSV's

    for file_settings in files:
        path = file_settings['path']
        start_frame = file_settings['start_frame']
        fps = file_settings['fps']
        interval = file_settings['samples']['interval']
        quantity = file_settings['samples']['quantity']
        csv_path = file_settings['samples']['csv']

        try:
            metadata = get_metadata(path)
            patches = sample_frames(path, metadata=metadata, fps=fps, interval=interval,
                                    quantity=quantity, start_frame=start_frame)
            save_patches(patches, csv_path)

        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise
        except Exception as e:
            raise

# CURRENTLY NOT WORKING!


def generate_curve(LUT_settings, files):
    LUT_size = LUT_settings['size']

    source_gamma = find_source_gamma(files)
    target_gamma = find_target_gamma(files)

    LUT_table = LUT3x1D.linear_table(LUT_size)
    source_grayscale, target_grayscale = load_patch_type(files, 'Grayscale')

    target_grayscale = cctf_encoding(cctf_decoding(
        target_grayscale, function=target_gamma), function=source_gamma)

    source_grayscale, target_grayscale = remove_saturated_patches(
        source_grayscale, target_grayscale)

    source_grayscale, target_grayscale = sort_patches(
        source_grayscale, target_grayscale)

    source_grayscale, target_grayscale = resample_patches(
        source_grayscale, target_grayscale)

    source_grayscale, target_grayscale = remove_outlier_patches(
        source_grayscale, target_grayscale)

    LUT_table = fit_curve(LUT_table, source_grayscale, target_grayscale)

    interpolation_results = np.array(
        [np.linspace(0, 1, 256), np.linspace(0, 1, 256), np.linspace(0, 1, 256)]).T

    from matplotlib import pyplot as plt
    plt.plot(LUT3x1D(np.clip(LUT_table, 0, 1)).apply(
        interpolation_results), interpolation_results, label='interp')
    plt.show()
    return LUT3x1D(np.clip(LUT_table, 0, 1))


def generate_matrix(LUT_settings, files, curve=None):
    LUT_size = LUT_settings['size']
    LUT_gamma = LUT_settings['gamma']
    LUT_algorithm = LUT_settings['algorithm']

    source_gamma = find_source_gamma(files)
    target_gamma = find_target_gamma(files)

    LUT_table = LUT3D.linear_table(LUT_size)

    source_colour, target_colour = load_patch_type(files, 'Colour')

    if curve != None:
        curve.apply(LUT_table)
        curve.apply(source_colour)

    LUT_table = cctf_decoding(LUT_table, function=source_gamma)

    LUT_table = solve_colour_matrix(
        LUT_table, source_colour, target_colour, source_gamma=source_gamma,
        target_gamma=target_gamma, LUT_gamma=LUT_gamma, algorithm=LUT_algorithm)

    LUT_table = cctf_encoding(LUT_table, function=LUT_gamma)

    return LUT3D(np.clip(LUT_table, 0, 1))


def create_LUT(LUT_settings, files):
    LUT_path = LUT_settings['output_file']

    curve = generate_curve(LUT_settings, files)
    LUT = generate_matrix(LUT_settings, files, curve)

    write_LUT(LUT, LUT_path, method='Resolve Cube')
    return curve


if __name__ == "__main__":
    LUT_settings, files = load_settings()
    # extract_patches(files)
    LUT = create_LUT(LUT_settings, files)
    plot_LUT(LUT)
