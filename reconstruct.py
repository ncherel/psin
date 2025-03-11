import numpy as np

from numba import njit, prange

from config import PATCH_SIZE, H_PATCH_SIZE
from utils import is_in_inner_boundaries


def reconstruct(src, shift_map, method="weighted"):
    """Reconstruct the video from the shift map.
    Only pixels in the occlusion are reconstructed
    use_all_patches: false during initialization
    """
    if method == "weighted":
        return weighted_reconstruction(src, shift_map)

    return best_patch_reconstruction(src, shift_map)


@njit
def best_patch_reconstruction(src, shift_map):
    """Best patch reconstruction. Given the weights of all surrounding patches
    Pick the colour of the best matching patch(patch with the lowest distance)
    """
    img = np.zeros((shift_map.shape[0], shift_map.shape[1], src.shape[2]), dtype=src.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i_min = max(i - H_PATCH_SIZE, 0)
            i_max = min(i + H_PATCH_SIZE, img.shape[0] - 1)
            j_min = max(j - H_PATCH_SIZE, 0)
            j_max = min(j + H_PATCH_SIZE, img.shape[1] - 1)

            # Check weights
            min_weights = 1e9
            avg_colour = np.zeros(img.shape[2], dtype=img.dtype)
            correct_info = False

            for ii in range(i_min, i_max + 1):
                for jj in range(j_min, j_max + 1):
                    if not is_in_inner_boundaries(img, ii, jj):
                        continue

                    if shift_map[ii, jj, 2] < min_weights:
                        min_weights = shift_map[ii, jj, 2]

                        ii_shift = i + int(shift_map[ii, jj, 0])
                        jj_shift = j + int(shift_map[ii, jj, 1])
                        avg_colour[:] = src[ii_shift, jj_shift, :]

                    correct_info = True

            if not correct_info:
                continue

            img[i, j, :] = avg_colour[:]

    return img


@njit(parallel=True)
def weighted_reconstruction(src, shift_map):
    """Compute a weighted average of the patches"""
    img = np.zeros((shift_map.shape[0], shift_map.shape[1], src.shape[2]), dtype=src.dtype)
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            weights = -1.0 * np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

            i_min = max(i - H_PATCH_SIZE, 0)
            i_max = min(i + H_PATCH_SIZE, img.shape[0] - 1)
            j_min = max(j - H_PATCH_SIZE, 0)
            j_max = min(j + H_PATCH_SIZE, img.shape[1] - 1)

            # Check weights
            max_weights = 1e-5
            correct_info = False

            valid_weights = []

            for ii in range(i_min, i_max + 1):
                for jj in range(j_min, j_max + 1):
                    if not is_in_inner_boundaries(img, ii, jj):
                        continue
                    weights[ii - i_min, jj - j_min] = shift_map[ii, jj, 2]
                    max_weights = max(weights[ii - i_min, jj - j_min], max_weights)
                    valid_weights.append(shift_map[ii, jj, 2])
                    correct_info = True

            if not correct_info:
                continue

            sum_weights = 0.0
            avg_colour = np.zeros(img.shape[2], dtype=img.dtype)

            sigma = max(1.0, np.percentile(valid_weights, 75))

            for ii in range(i_min, i_max + 1):
                for jj in range(j_min, j_max + 1):
                    if not is_in_inner_boundaries(img, ii, jj):
                        continue

                    # Apply (ii, jj, kk) shift to pixel (i,j)
                    ii_shift = i + int(shift_map[ii, jj, 0])
                    jj_shift = j + int(shift_map[ii, jj, 1])

                    # Adapt the weights
                    weights[ii - i_min, jj - j_min] = np.exp(-weights[ii - i_min, jj - j_min] / (2 * sigma**2))
                    sum_weights += weights[ii - i_min, jj - j_min]

                    # Weighted sum
                    for c in range(img.shape[2]):
                        avg_colour[c] += weights[ii - i_min, jj - j_min] * src[ii_shift, jj_shift, c]

            img[i, j, :] = avg_colour / sum_weights

    return img
