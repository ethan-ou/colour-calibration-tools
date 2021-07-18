import matplotlib.pyplot as plt
from utils import scale_range_1, scale_range_255, split_to_RGB
from extract import load_patches
import numpy as np
from colour import LUT3D, read_LUT, cctf_encoding, RGB_to_RGB, RGB_COLOURSPACES


def plot_samples(colours):
    r, g, b = scale_range_255(split_to_RGB(colours))

    ax = plt.axes(projection='3d')

    ax.scatter(r, g, b, c=np.array(colours))
    plt.show()


def plot_LUT(LUT, size=12, gamma=None, colourspace=None):
    samples = LUT3D.linear_table(size)

    if gamma:
        samples = cctf_encoding(samples, function=gamma)
    if colourspace:
        samples = RGB_to_RGB(
            samples, input_colourspace=RGB_COLOURSPACES['ITU-R BT.709'], output_colourspace=RGB_COLOURSPACES[colourspace])

    samples = np.clip(LUT.apply(np.reshape(samples, (-1, 3))), 0, 1)
    r, g, b = scale_range_255(split_to_RGB(samples))

    ax = plt.axes(projection='3d')
    ax.scatter(r, g, b, c=samples)
    plt.show()
