import cv2
import numpy as np

from os.path import join

from utils import interpolate_shift_map, get_pyramid
from patchmatch import patch_match
from reconstruct import reconstruct
from initialisation import initialisation, initialisation_from_img

from config import (USE_TEXTURE, BETA, MAX_ITER, RESIDUAL_THRESH)


def get_texture(img):
    """Compute the texture features (gradients).
    Only the magnitude of the gradient in each direction is kept (|gx|, |gy|).
    returns: (H, W, 2)
    """
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gy, gx = np.gradient(grayscale)

    gx = np.abs(gx)
    gy = np.abs(gy)

    return np.stack((gy, gx), axis=-1)


def generate(src, init=None, nb_levels=3, output="."):
    print("Computing features and pyramid...", end=" ", flush=True)
    if USE_TEXTURE:
        texture = get_texture(src)
        src = np.concatenate((src, BETA * texture), axis=-1)
    src_pyramid = get_pyramid(src, nb_levels=nb_levels)
    print("Done")

    patch_match_params = {
        "n_iters": 10,
        "alpha": 0.5,
        "w": max(src.shape[:2]),
    }

    print("Initialisation...", end=" ", flush=True)
    if init is not None:
        img, shift_map = initialisation_from_img(src_pyramid[-1], init, patch_match_params)
    else:
        img, shift_map = initialisation(src_pyramid[-1], patch_match_params)
    print("Done")

    for level in reversed(range(nb_levels)):
        print(f"Level {level}...", end=" ", flush=True)
        src = src_pyramid[level]

        # Interpolate the shift volume and reconstruct at this level
        if level != (nb_levels - 1):
            shift_map = interpolate_shift_map(shift_map, src.shape)
            img = reconstruct(src, shift_map, method="weighted")

        iteration_nb = 1
        residual = float("inf")
        while iteration_nb <= MAX_ITER and residual > RESIDUAL_THRESH:
            previous_img = img.copy()

            shift_map = patch_match(img, src, shift_map, patch_match_params)
            img = reconstruct(src, shift_map, method="weighted")

            iteration_nb += 1
            residual = np.mean(np.abs(img - previous_img))

        print("Done")
        cv2.imwrite(join(output, f"iter_{level}.png"), img[..., :3] * 255)


if __name__ == '__main__':
    src = cv2.imread("balloons.png").astype(np.float32) / 255.0
    generate(src, nb_levels=4)
