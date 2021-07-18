from utils import RGB_to_original_shape, split_to_RGB
import numpy as np
from colour.characterisation import polynomial_expansion_Finlayson2015


def root_polynomial_colour_correction(RGB, matrix, degree=3):
    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))

    RGB_e = polynomial_expansion_Finlayson2015(RGB, degree,
                                               root_polynomial_expansion=True)

    return np.reshape(np.transpose(np.dot(matrix, np.transpose(RGB_e))), shape)


def iter_tetrahedral_colour_correction(RGB, matrix):
    def tetra_inner(RGB, matrix):
        r, g, b = RGB

        wht = np.array([1, 1, 1])
        red, yel, grn, cyn, blu, mag = matrix

        if r > g:
            if g > b:
                return r*red + g*(yel-red) + b*(wht-yel)
            elif r > b:
                return r*red + g*(wht-mag) + b*(mag-red)
            else:
                return r*(mag-blu) + g*(wht-mag) + b*blu
        else:
            if b > g:
                return r*(wht-cyn) + g*(cyn-blu) + b*blu
            elif b > r:
                return r*(wht-cyn) + g*grn + b*(cyn-grn)
            else:
                return r*(yel-grn) + g*grn + b*(wht-yel)

    shape = RGB.shape

    RGB = np.reshape(RGB, (-1, 3))
    RGB = np.array([tetra_inner(colour, matrix) for colour in RGB])
    return np.array(RGB).reshape(shape)


def tetrahedral_colour_correction(RGB, matrix):
    # Return indicies of boolean comparison
    # e.g. a = [0, 1, 3], b = [1, 1, 1]
    # i((a > b)) -> [1, 2]
    def i(arr):
        return arr.nonzero()[0]

    # Find and remove existing elements (emulates if statement)
    # e.g. exists([0, 1, 2], [[1], [0]]) -> [2]
    def exists(arr, prev):
        return np.setdiff1d(arr, np.concatenate(prev))

    # RGB Multiplication of Tetra
    def t_matrix(index, r, mult_r, g, mult_g, b, mult_b):
        return np.multiply.outer(r[index], mult_r) + np.multiply.outer(g[index], mult_g) + np.multiply.outer(b[index], mult_b)

    shape = RGB.shape
    RGB = np.reshape(RGB, (-1, 3))
    r, g, b = RGB.T

    wht = np.array([1, 1, 1])
    red, yel, grn, cyn, blu, mag = matrix

    base_1 = r > g
    base_2 = ~(r > g)

    case_1 = i(base_1 & (g > b))
    case_2 = exists(i(base_1 & (r > b)), [case_1])
    case_3 = exists(i(base_1), [case_1, case_2])
    case_4 = i(base_2 & (b > g))
    case_5 = exists(i(base_2 & (b > r)), [case_4])
    case_6 = exists(i(base_2), [case_4, case_5])

    n_RGB = np.zeros(RGB.shape)
    n_RGB[case_1] = t_matrix(case_1, r, red, g, (yel-red), b, (wht-yel))
    n_RGB[case_2] = t_matrix(case_2, r, red, g, (wht-mag), b, (mag-red))
    n_RGB[case_3] = t_matrix(case_3, r, (mag-blu), g, (wht-mag), b, blu)
    n_RGB[case_4] = t_matrix(case_4, r, (wht-cyn), g, (cyn-blu), b, blu)
    n_RGB[case_5] = t_matrix(case_5, r, (wht-cyn), g, grn, b, (cyn-grn))
    n_RGB[case_6] = t_matrix(case_6, r, (yel-grn), g, grn, b, (wht-yel))

    return n_RGB.reshape(shape)


# Add 0 and 1 points as an option
def curve_colour_correction(RGB, source_points, target_points):
    def gauss_jordan_solve(A):
        m = len(A)

        for k in range(m):
            i_max = 0
            vali = None

            for i in range(k, m):
                if vali == None or abs(A[i][k]) > vali:
                    i_max = i
                    vali = abs(A[i][k])

            [A[k], A[i_max]] = [A[i_max], A[k]]

            for i in range(k + 1, m):
                cf = A[i][k] / A[k][k]

                for j in range(k, m + 1):
                    A[i][j] -= A[k][j] * cf

        x = []

        for i in range(m - 1, -1, -1):
            # rows = columns
            v = A[i][m] / A[i][i]

            # Fills values with 0 if list not initialised
            if len(x) == 0:
                x = np.zeros(i + 1)

            x[i] = v

            for j in range(i-1, -1, -1):
                # rows
                A[j][m] -= A[j][i] * v
                A[j][i] = 0

        return x

    def get_natural_ks(xs, ys):
        n = len(xs) - 1
        A = np.zeros((n + 1, n + 2))

        for i in range(1, n):
            A[i][i - 1] = 1 / (xs[i] - xs[i - 1])

            A[i][i] = 2 * (1 / (xs[i] - xs[i - 1]) +
                           1 / (xs[i + 1] - xs[i]))
            A[i][i + 1] = 1 / (xs[i + 1] - xs[i])
            A[i][n + 1] = 3 * ((ys[i] - ys[i - 1]) / ((xs[i] - xs[i - 1]) * (xs[i] - xs[i - 1])) +
                               (ys[i + 1] - ys[i]) / ((xs[i + 1] - xs[i]) * (xs[i + 1] - xs[i])))

        A[0][0] = 2 / (xs[1] - xs[0])
        A[0][1] = 1 / (xs[1] - xs[0])
        A[0][n + 1] = (3 * (ys[1] - ys[0])) / \
            ((xs[1] - xs[0]) * (xs[1] - xs[0]))

        A[n][n - 1] = 1 / (xs[n] - xs[n - 1])
        A[n][n] = 2 / (xs[n] - xs[n - 1])
        A[n][n + 1] = (3 * (ys[n] - ys[n - 1])) / \
            ((xs[n] - xs[n - 1]) * (xs[n] - xs[n - 1]))

        return gauss_jordan_solve(A)

    def eval_spline(x, xs, ys, ks):
        i = 1
        print(xs)
        while xs[i] < x:
            i = i+1

        ks = ks

        t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        a = ks[i - 1] * (xs[i] - xs[i - 1]) - (ys[i] - ys[i - 1])
        b = -ks[i] * (xs[i] - xs[i - 1]) + (ys[i] - ys[i - 1])

        return (1 - t) * ys[i - 1] + t * ys[i] + t * (1 - t) * (a * (1 - t) + b * t)

    shape = RGB.shape

    red, green, blue = np.clip(split_to_RGB(RGB), 0, 1)

    black_point = np.array([[0], [0], [0]])
    white_point = np.array([[1], [1], [1]])

    source_points = np.concatenate(
        [black_point, source_points, white_point], axis=1)
    target_points = np.concatenate(
        [black_point, target_points, white_point], axis=1)

    s_red, s_green, s_blue = source_points
    t_red, t_green, t_blue = target_points

    def v_eval_spline(channel, source, target):
        ks = get_natural_ks(source, target)

        return [eval_spline(val, source, target, ks) for val in channel]

    RGB = np.array([v_eval_spline(red, s_red, t_red), v_eval_spline(
        green, s_green, t_green), v_eval_spline(blue, s_blue, t_blue)])

    return RGB_to_original_shape(RGB, shape)
