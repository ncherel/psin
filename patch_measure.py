import numpy as np

from cffi import FFI
from numba import njit

from config import PATCH_SIZE, H_PATCH_SIZE, USE_CPP


if USE_CPP:
    ffi = FFI()
    ffi.cdef("""
    float patch_measure(float* img1, float* img2,
		    int x1_size, int x2_size, int c_size,
		    int y_a, int x_a,
		    int y_b, int x_b,
		    float min_val);
    """)
    C = ffi.dlopen("./libpatch_measure.so")
    patch_measure_cpp = C.patch_measure


@njit
def calculate_patch_distance(img1, img2, shift_map, y_a, x_a):
    """Compute the patch distance at a given position between a patch and
    its matching patch
    """
    y_b = y_a + int(shift_map[y_a, x_a, 0])
    x_b = x_a + int(shift_map[y_a, x_a, 1])

    return patch_measure(img1, img2, y_a, x_a, y_b, x_b, 1e10)


@njit
def patch_measure(img1, img2, y_a, x_a, y_b, x_b, min_val):
    if USE_CPP:
        return patch_measure_cpp(ffi.from_buffer(img1), ffi.from_buffer(img2),
                                 img1.shape[1], img2.shape[1], img1.shape[2],
                                 y_a, x_a,
                                 y_b, x_b,
                                 min_val)

    return patch_measure_numba(img1, img2, y_a, x_a, y_b, x_b, min_val)


@njit
def patch_measure_numba(img1, img2, y_a, x_a, y_b, x_b, min_val):
    """Compute the distance between two patches"""
    distance = 0.0
    for i in range(-H_PATCH_SIZE, H_PATCH_SIZE + 1):
        for j in range(-H_PATCH_SIZE, H_PATCH_SIZE + 1):
            for c in range(img1.shape[2]):
                diff = img1[y_a + i, x_a + j, c] - img2[y_b + i, x_b + j, c]
                distance += (diff * diff)

        # check if the patch distance has not exceeded the current best one
        if distance > min_val:
            return distance

    return distance
