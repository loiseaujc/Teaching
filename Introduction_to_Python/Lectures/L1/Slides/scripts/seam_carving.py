import numpy as np

from scipy.ndimage.filters import convolve
from imageio import imread, imwrite

import numba

import matplotlib.pyplot as plt

from tqdm import trange

def compute_energy(img):

    # --> Convolution kernel.
    kernel = np.array([
    [1.0, 2.0, 1.0],
    [0.0, 0.0, 0.0],
    [-1.0, -2.0, -1.0]
    ])

    # --> Horizontal kernel
    kernel_x = np.stack([kernel]*3, axis=2)

    # --> Vertical kernel.
    kernel_y = np.stack([kernel.T]*3, axis=2)

    # --> Convolved image.
    img = img.astype("float32")
    conv_img = np.abs(convolve(img, kernel_x)) + np.abs(convolve(img, kernel_y))

    # --> Energy map.
    energy_map = conv_img.sum(axis=2)

    return energy_map

def seam_carve(img, scale=0.5):
    r, c, _ = img.shape
    new_c = int(scale * c)

    for i in trange(c - new_c):
        img = carve_column(img)

    return img

def carve_column(img):

    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask]*3, axis=2)
    img = img[mask].reshape((r, c-1, 3))

    return img

@numba.jit()
def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = compute_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):

            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def plot_img(img, shape=None):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.imshow(img)

    ax.set_aspect("equal")
    ax.axis("off")

    if shape is not None:
        ax.set_xlim(0, shape[0])
        ax.set_ylim(shape[1], 0)

    return fig, ax

def main():

    fname = "raw_img.jpg"
    img = imread(fname)
    imwrite("../imgs/raw_img.jpg", img)

    scales = [0.5, 0.6, 0.7, 0.8, 0.9]

    for i, scale in enumerate(scales):
        out_img = seam_carve(img, scale=scale)
        fname = "../imgs/cropped_img_%04i.jpg" % i
        imwrite(fname, out_img)

if __name__ == "__main__":
    main()
