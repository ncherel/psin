import cv2
import numpy as np

from numba import njit
from scipy.ndimage.filters import gaussian_filter

from config import H_PATCH_SIZE, SUBSAMPLING


@njit
def is_in_inner_boundaries(img, y, x):
    return ((H_PATCH_SIZE <= y < img.shape[0] - H_PATCH_SIZE)
            and (H_PATCH_SIZE <= x < img.shape[1] - H_PATCH_SIZE))

@njit
def is_same_shift(current_shift, shift_y, shift_x):
    """Check if the candidate shift is the same as the current one. This
    avoids recomputing the distance
    """
    return ((current_shift[0] == shift_y)
            and (current_shift[1] == shift_x))


def get_pyramid(img, nb_levels, sigma=1.5):
    """Compute the img pyramid for levels from 1 to nb_levels.
    Preserves temporal and channel dimensions
    level 0 (H, W, C)
    level 1 (H/2, W/2, C)
    level 2 (H/4, W/4, C)
    ...
    """
    img_pyramid = [img]
    previous_level = img

    for lvl in range(1, nb_levels):
        # set up new spatial sizes
        new_y_size = len(np.arange(0, previous_level.shape[0], SUBSAMPLING))
        new_x_size = len(np.arange(0, previous_level.shape[1], SUBSAMPLING))
        lvl_img = np.zeros((new_y_size, new_x_size, img.shape[2]), dtype=img.dtype)

        for c in range(img.shape[2]):
            frame = gaussian_filter(previous_level[..., c], sigma, mode="mirror", truncate=0.75)
            lvl_img[..., c] = frame[::SUBSAMPLING, ::SUBSAMPLING]

        img_pyramid.append(lvl_img)
        previous_level = lvl_img

    return img_pyramid


@njit
def nclip(low, a, high):
    if a >= high:
        return high
    if a <= low:
        return low
    return a



@njit
def interpolate_shift_map(shift_map_in, shape):
    """Interpolate the shift map with economy of memory and speed
    Only interpolate for the valid patches [H_PATCH_SIZE, -H_PATCH_SIZE]
    """
    shift_map_out = 10000 * np.ones((shape[0], shape[1], 3), dtype=shift_map_in.dtype)

    cy = (shift_map_in.shape[0] - 2 * H_PATCH_SIZE - 1) / (shape[0] - 2 * H_PATCH_SIZE - 1)
    cx = (shift_map_in.shape[1] - 2 * H_PATCH_SIZE - 1) / (shape[1] - 2 * H_PATCH_SIZE - 1)
    a = H_PATCH_SIZE - np.round(H_PATCH_SIZE * cy)

    for i in range(shape[0]):
        for j in range(shape[1]):

            # Skip invalid positions
            if not is_in_inner_boundaries(shift_map_out, i, j):
                continue

            ii = int(np.round(cy * i) + a)
            jj = int(np.round(cx * j) + a)

            shift_map_out[i, j, 0] = nclip(H_PATCH_SIZE - i, 2 * shift_map_in[ii, jj, 0], shape[0] - 1 - H_PATCH_SIZE - i)
            shift_map_out[i, j, 1] = nclip(H_PATCH_SIZE - j, 2 * shift_map_in[ii, jj, 1], shape[1] - 1 - H_PATCH_SIZE - j)
            shift_map_out[i, j, 2] = shift_map_in[ii, jj, 2]

    return shift_map_out
