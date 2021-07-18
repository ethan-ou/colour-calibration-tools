import numpy as np
from scipy.signal import resample
from settings import find_curr_batch_files, find_num_batches
from utils import RGB_to_original_shape, split_to_RGB
from extract import load_patches
from colour import cctf_decoding, cctf_encoding, sRGB_to_XYZ, XYZ_to_Lab, delta_E
from colour.characterisation import colour_correction_Finlayson2015, matrix_colour_correction_Finlayson2015
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from colour_matrix import curve_colour_correction, root_polynomial_colour_correction, tetrahedral_colour_correction


def load_patch_type(files, patch_type):
    source_patches = []
    target_patches = []

    for batch_num in range(1, find_num_batches(files) + 1):
        source, target = find_curr_batch_files(files, batch_num, patch_type)

        if source == None or target == None:
            break

        source_csv = load_patches(source['samples']['csv'])
        target_csv = load_patches(target['samples']['csv'])

        source_patches.extend(source_csv)
        target_patches.extend(target_csv)

    return (np.array(source_patches), np.array(target_patches))


def linearise_colour_patches(source_patches, target_patches, source_gamma, target_gamma):
    return (cctf_decoding(source_patches, function=source_gamma), cctf_decoding(target_patches, function=target_gamma))


def gamma_encode_colour_patches(source_patches, target_patches, source_gamma, target_gamma):
    return (cctf_encoding(source_patches, function=source_gamma), cctf_encoding(target_patches, function=target_gamma))


def fit_curve(values, source_patches, target_patches):
    def smooth_RGB(RGB):
        window_length = 9  # Magic number
        poly_order = 3  # Currently cubic order

        red, green, blue = split_to_RGB(RGB)
        smooth = np.array((savgol_filter(red, window_length, poly_order), savgol_filter(
            green, window_length, poly_order), savgol_filter(blue, window_length, poly_order)))

        return smooth.T

    source_red, source_green, source_blue = smooth_RGB(source_patches).T
    target_red, target_green, target_blue = smooth_RGB(target_patches).T

    # source_red, target_red = sort_channel(source_red, target_red)
    # source_green, target_green = sort_channel(source_green, target_green)
    # source_blue, target_blue = sort_channel(source_blue, target_blue)

    source_points = np.array([source_red, source_green, source_blue])
    target_points = np.array([target_red, target_green, target_blue])

    return smooth_RGB(curve_colour_correction(values, source_points, target_points))


def solve_colour_matrix(values, source_patches, target_patches, source_gamma, target_gamma, LUT_gamma, algorithm='Root-Polynomial'):
    tetrahedral_matrix = np.array(
        [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]])
    tetrahedral_matrix_shape = tetrahedral_matrix.shape
    tetrahedral_matrix_weights = tetrahedral_matrix.flatten()
    tetrahedral_matrix_length = tetrahedral_matrix_weights.shape

    root_polynomial_matrix = matrix_colour_correction_Finlayson2015(
        source_patches, target_patches, degree=3, root_polynomial_expansion=True)
    root_polynomial_shape = root_polynomial_matrix.shape
    root_polynomial_weights = root_polynomial_matrix.flatten()
    root_polynomial_length = root_polynomial_weights.shape

    def solve_fn(matrix, source, target, degree=3):
        s_tetra_matrix = matrix
        # s_tetra_matrix = np.split(matrix, tetrahedral_matrix_length)[0]
        # s_root_polynomial_matrix = np.split(
        #     matrix, tetrahedral_matrix_length)[1]

        source = tetrahedral_colour_correction(
            source, s_tetra_matrix.reshape(tetrahedral_matrix_shape))

        # source = np.clip(root_polynomial_colour_correction(
        #     source, s_root_polynomial_matrix.reshape(root_polynomial_shape), degree), 0, 1000)

        source, target = gamma_encode_colour_patches(
            source, target, LUT_gamma, LUT_gamma)

        delta_E = np.sum(find_delta_E(source, target))
        print(delta_E)

        return delta_E

    source_patches, target_patches = linearise_colour_patches(
        source_patches, target_patches, source_gamma, target_gamma)

    source_patches, target_patches = remove_saturated_patches(
        source_patches, target_patches)

    # weights = np.concatenate(
    #     (tetrahedral_matrix_weights, root_polynomial_weights), axis=None)

    solve = least_squares(solve_fn, tetrahedral_matrix_weights, args=(
        source_patches, target_patches))

    # tetrahedral_matrix = np.reshape(np.split(solve.x, tetrahedral_matrix_length)[
    #                                 0], tetrahedral_matrix_shape)
    # root_polynomial_matrix = np.reshape(np.split(solve.x, tetrahedral_matrix_length)[
    #                                     1], root_polynomial_shape)

    # root_polynomial_matrix = np.reshape(solve.x, root_polynomial_shape)
    tetrahedral_matrix = np.reshape(solve.x, tetrahedral_matrix_shape)
    # {
    #     'Root-Polynomial': np.clip(colour_correction_Finlayson2015(values, source_patches, target_patches, degree=3, root_polynomial_expansion=True), 0, 1000),
    #     'Linear Matrix': np.clip(colour_correction_Finlayson2015(values, source_patches, target_patches, degree=1), 0, 1000)
    # }[algorithm]

    return tetrahedral_colour_correction(values, tetrahedral_matrix)


def sort_channel(source_channel, target_channel):
    sorted_idx = source_channel.argsort()

    return (source_channel[sorted_idx], target_channel[sorted_idx])


def sort_patches(source_patches, target_patches):
    source_shape = np.array(source_patches).shape
    target_shape = np.array(target_patches).shape

    source_red, source_green, source_blue = split_to_RGB(source_patches)
    target_red, target_green, target_blue = split_to_RGB(target_patches)

    # Sort based on Source 'Green' channel
    sort_r = source_red.argsort()
    sort_g = source_green.argsort()
    sort_b = source_blue.argsort()

    source_RGB = np.array([
        source_red[sort_r], source_green[sort_g], source_blue[sort_b]]).T.reshape(source_shape)
    target_RGB = np.array([
        target_red[sort_r], target_green[sort_g], target_blue[sort_b]]).T.reshape(target_shape)

    return (source_RGB, target_RGB)


def remove_saturated_patches(source, target, max_value=1):
    source_allowed = np.where(
        ((source > 0) & (source < max_value)).all(axis=1))
    target_allowed = np.where(
        ((target > 0) & (target < max_value)).all(axis=1))

    indicies = np.intersect1d(source_allowed, target_allowed)

    return (np.take(source, indicies, axis=0), np.take(target, indicies, axis=0))


def resample_patches(source, target, max_samples=24):
    source_shape = np.array(source).shape
    target_shape = np.array(target).shape

    s_red, s_green, s_blue = split_to_RGB(source)
    t_red, t_green, t_blue = split_to_RGB(target)

    if s_green.shape[0] > max_samples or t_green.shape[0] > max_samples:
        s_red = resample(s_red, max_samples)
        t_red = resample(t_red, max_samples)
        s_green = resample(s_green, max_samples)
        t_green = resample(t_green, max_samples)
        s_blue = resample(s_blue, max_samples)
        t_blue = resample(t_blue, max_samples)

    return (RGB_to_original_shape(np.array([s_red, s_green, s_blue]), (-1, source_shape[1])),
            RGB_to_original_shape(np.array([t_red, t_green, t_blue]), (-1, target_shape[1])))


def remove_outlier_patches(source, target, max_delta_E=10):
    delta_E = find_delta_E(source, target)
    allowed = np.where(delta_E < max_delta_E)

    return (source[allowed], target[allowed])


def find_delta_E(source, target):
    return delta_E(XYZ_to_Lab(sRGB_to_XYZ(source)),
                   XYZ_to_Lab(sRGB_to_XYZ(target)))
