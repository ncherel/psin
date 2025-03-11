import numpy as np

from numpy.random import randint
from numba import njit, prange

from utils import is_in_inner_boundaries, is_same_shift
from patch_measure import patch_measure, calculate_patch_distance

from config import H_PATCH_SIZE


@njit
def is_valid_match(img, i, j):
    """Check if the match is valid (inside)"""
    return is_in_inner_boundaries(img, i, j)


@njit
def sample_around(x, window, low=None, high=None):
    """Sample a point in [max(x - window, low), min(x + window, high)]"""
    rand_min = max(x - window, low)
    rand_max = min(x + window + 1, high)

    if rand_min >= rand_max:
        return int(rand_min)

    return randint(rand_min, rand_max)


def patch_match(img1, img2, first_guess, params):
    """Compute a shift map from img1 to img2"""
    shift_map = initialise_shift_map(first_guess, img1, img2)

    for i in range(params["n_iters"]):
        propagation(shift_map, img1, img2, i)
        random_search(shift_map, img1, img2, params["w"], params["alpha"])

    return shift_map


@njit(parallel=True)
def initialise_shift_map(shift_map, img1, img2):
    """Initialise the shift map by sampling valid coordinates. If the
    existing shift is valid, update the distance.
    """
    for i in prange(img1.shape[0]):
        for j in prange(img1.shape[1]):
            if not is_in_inner_boundaries(img1, i, j):
                continue
            y_shift = int(shift_map[i, j, 0])
            x_shift = int(shift_map[i, j, 1])

            while not is_valid_match(img2, i + y_shift, j + x_shift):
                y_shift = randint(H_PATCH_SIZE, img2.shape[0] - H_PATCH_SIZE) - i
                x_shift = randint(H_PATCH_SIZE, img2.shape[1] - H_PATCH_SIZE) - j

            shift_map[i, j, 0] = float(y_shift)
            shift_map[i, j, 1] = float(x_shift)
            shift_map[i, j, 2] = calculate_patch_distance(img1, img2, shift_map, i, j)

    return shift_map


@njit
def propagation(shift_map, img1, img2, iteration_nb):
    """Propagation step that evaluate neighbors shift"""
    if iteration_nb % 2 == 0:
        shift = -1
        irange = range(img1.shape[0])
        jrange = range(img1.shape[1])
    else:
        shift = 1
        irange = range(img1.shape[0]-1, -1, -1)
        jrange = range(img1.shape[1]-1, -1, -1)

    for i in irange:
        for j in jrange:
            if not is_in_inner_boundaries(img1, i, j):
                continue

            current_shift = shift_map[i, j, :2]
            current_distance = shift_map[i, j, 2]

            neighbors = [[i + shift, j],  # up / down
                         [i, j + shift]]  # left / right

            for (y, x) in neighbors:
                if not is_in_inner_boundaries(img1, y, x):
                    continue

                y_shift = int(shift_map[y, x, 0])
                x_shift = int(shift_map[y, x, 1])

                if not is_valid_match(img2, i + y_shift, j + x_shift):
                    continue

                if is_same_shift(current_shift, y_shift, x_shift):
                    continue

                distance = patch_measure(img1, img2,
                                         i, j,
                                         i + y_shift, j + x_shift,
                                         current_distance)

                if distance < current_distance:
                    current_shift[:] = [y_shift, x_shift]
                    current_distance = distance

            if current_distance < shift_map[i, j, 2]:
                shift_map[i, j, :2] = current_shift
                shift_map[i, j, 2] = current_distance


@njit(parallel=True)
def random_search(shift_map, img1, img2, max_window, alpha):
    """Randomly search around each match to find better matches"""
    max_window = min(max_window, max(img2.shape[:-1]))
    max_exponent = int(np.ceil(-np.log(max_window) / np.log(alpha)))

    windows = np.zeros(max_exponent, dtype=np.int64)
    for exponent in range(max_exponent):
        windows[exponent] = max_window * np.power(alpha, exponent)

    for i in prange(img1.shape[0]):
        for j in prange(img1.shape[1]):
            if not is_in_inner_boundaries(img1, i, j):
                continue

            current_shift_x = int(shift_map[i, j, 0])
            current_shift_y = int(shift_map[i, j, 1])
            current_shift = [current_shift_x, current_shift_y]
            current_distance = shift_map[i, j, 2]

            y_match = i + current_shift_x
            x_match = j + current_shift_y

            for window in windows:
                y_rand = sample_around(y_match, window, H_PATCH_SIZE, img2.shape[0] - H_PATCH_SIZE)
                x_rand = sample_around(x_match, window, H_PATCH_SIZE, img2.shape[1] - H_PATCH_SIZE)

                if not is_valid_match(img2, y_rand, x_rand):
                    continue

                if is_same_shift(current_shift, y_rand - i, x_rand - j):
                    continue

                distance = patch_measure(img1, img2,
                                         i, j,
                                         y_rand, x_rand,
                                         current_distance)

                if distance < current_distance:
                    current_shift[:] = [y_rand - i, x_rand - j]
                    current_distance = distance

            if current_distance < shift_map[i, j, 2]:
                shift_map[i, j, 0] = float(current_shift[0])
                shift_map[i, j, 1] = float(current_shift[1])
                shift_map[i, j, 2] = current_distance
