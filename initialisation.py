import numpy as np
import cv2

from patchmatch import patch_match
from reconstruct import reconstruct

from config import PATCH_SIZE, BETA


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


def initialisation(src, patch_match_params):
    img = np.random.randn(src.shape[0], src.shape[1], 3).astype(np.float32) + 0.5
    shift_map = 10000 * np.ones((src.shape[0], src.shape[1], 3), dtype=np.float32)

    no_texture_src = np.copy(src[..., :3])

    shift_map = patch_match(img, no_texture_src, shift_map, patch_match_params)
    img = reconstruct(src, shift_map, method="weighted")

    return img, shift_map


def initialisation_from_img(src, init, patch_match_params):
    # Extract features and resize
    texture = get_texture(init)
    init = np.concatenate((init, BETA * texture), axis=-1)

    img = np.zeros_like(src)
    for c in range(src.shape[2]):
        img[..., c] = cv2.resize(init[..., c], (src.shape[1], src.shape[0]), cv2.INTER_NEAREST)

    shift_map = 10000 * np.ones((src.shape[0], src.shape[1], 3), dtype=np.float32)

    shift_map = patch_match(img, src, shift_map, patch_match_params)
    img = reconstruct(src, shift_map, method="weighted")

    return img, shift_map
