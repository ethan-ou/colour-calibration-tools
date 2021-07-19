import numpy as np


def scale_range_1(values, prev_range=255):
    return np.array(values).astype('float64') / prev_range


def scale_range_255(values, prev_range=1):
    return np.array(values).astype('float64') * (255 / prev_range)


def split_to_RGB(values):
    return np.reshape(np.array(values), (-1, 3)).T


def RGB_to_original_shape(values, shape):
    return np.reshape(np.array(values).T, shape)
