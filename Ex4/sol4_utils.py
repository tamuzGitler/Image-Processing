################################ Imports ####################################

import imageio
from scipy.signal import convolve2d
import numpy as np
from scipy.ndimage.filters import convolve
import skimage.color

################################ Constants ####################################

MAX_GREY_VALUE = 255
GRAYSCALE = 1


################################ Functions ####################################

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


################################ prev ex functions ########################
def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imageio.imread(filename)  # to run write imageio.v2.imread

    if ((image.ndim == 3) and (representation == GRAYSCALE)):  # added dim check
        image = skimage.color.rgb2gray(image)

    else:  # RGB CASE
        image = image / MAX_GREY_VALUE  # normalize
    return image


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    # pyr == G in slides
    pyr, filter_vec = [im], create_filter_vec(filter_size)
    i = 1
    while i < max_levels and shape_not_smaller_than_16(pyr[i - 1]):
        reduced = reduce(pyr[i - 1], filter_vec)
        pyr.append(reduced)
        i += 1
    return pyr, filter_vec


################################ prev ex Helpers ###############################
def create_filter_vec(filter_size):
    """
    creates filter vector in size of filter_size using convolution with [1,1]
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    to be used in constructing the pyramid filter
    :return: filter_vec
    """
    if filter_size == 1:
        return np.array([1])
    elif filter_size == 2:
        return np.array([1, 1])
    filter_vec = np.ones(2)
    one_vec = filter_vec
    for i in range(2, filter_size):
        filter_vec = np.convolve(one_vec, filter_vec)
    return (filter_vec / filter_vec.sum()).reshape([1, filter_vec.shape[0]])  # normalized and reshaped!


def shape_not_smaller_than_16(im):
    """
    check im shape
    :param im: image
    :return: true if in good shape false otherwise
    """
    return im.shape[0] > 16 and im.shape[1] > 16


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blur_rows = convolve(im, blur_filter)  # blur rows
    blur_im = convolve(blur_rows, blur_filter.T)  # blur cols
    reduced_image = blur_im[::2, :][:, ::2]  # select odds rows and cols - reduces image size
    return reduced_image
