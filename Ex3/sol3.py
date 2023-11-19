################################ Imports ####################################

import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.ndimage.filters import convolve

################################ Constants ####################################

GRAYCOLOR = "gray"
MAX_GREY_VALUE = 255
GRAYSCALE = 1

################################ Implement Functions ##########################
######## Base Functions ########

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


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    expand_image = np.zeros((len(im) * 2, len(im[0]) * 2))  # *2 to maintain constant brightness
    expand_image[::2, :][:, ::2] = im  # every second pixel will be taken from im
    blur_rows = convolve(expand_image, 2 * blur_filter)  # blur rows
    expand_blur_image = convolve(blur_rows, 2 * blur_filter.T)  # blur cols, *2 to maintain constant brightness
    return expand_blur_image


######## Task 3.1 ########

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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
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
    # pyr == L == (Gi - Expand(Gi+1)) from slides
    gausian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    max_levels = min(len(gausian_pyr), max_levels) - 1  # prevent index out of bound in gausian_pyr[i + 1]
    for i in range(max_levels):
        pyr.append(np.asarray(gausian_pyr[i]) - expand(gausian_pyr[i + 1], filter_vec))
    pyr.append(gausian_pyr[-1])  # add to the end the last gausian pyramid
    return pyr, filter_vec


######## Task 3.1 Helper ########
def shape_not_smaller_than_16(im):
    """
    check im shape
    :param im: image
    :return: true if in good shape false otherwise
    """
    return im.shape[0] > 16 and im.shape[1] > 16


######## Task 3.2  ########

def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    # Before reconstructing multiply each level i of the laplacian pyramid by its corresponding coefficient coeff[i].
    gausian = lpyr[-1]  # init with  last laplacian image - Ln=Gn
    for i in range(len(lpyr)):
        lpyr[i] *= coeff[i]

    for i in reversed(range(len(lpyr) - 1)):  # run on the lpyr backwards
        gausian = expand(gausian, filter_vec) + lpyr[i]  # using exapnd on prev result to get same shape as lpyr[i ]
    return gausian


######## Task 3.3  ########


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    col_len, start_row, start_col = 0, 0, 0
    for i in range(levels):
        col_len += pyr[i].shape[1]
    res = np.zeros([pyr[0].shape[0], col_len])
    for i in range(levels):
        end_row = pyr[i].shape[0]
        end_col = start_col + pyr[i].shape[1]
        res[start_row:end_row, start_col:end_col] = (pyr[i] - pyr[i].min()) / (
                pyr[i].max() - pyr[i].min())  # stretch the values of each pyramid level to [0, 1]
        start_col = end_col  # update column boundry for next iteration
    return res


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap=GRAYCOLOR)
    plt.show()


######## Task 4  ########

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    # pyramid in small index contains high frequency's, high index contains low frequency's
    L1, L1FilterVec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, L2FilterVec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gaussianMask, gaussianMaskFilter = build_gaussian_pyramid(mask.astype(np.float64), max_levels,
                                                              filter_size_mask)  # float64 - fractional values should appear while constructing the mask’s pyramid

    Lc = (np.asarray(gaussianMask) * np.asarray(L1) + (1 - np.asarray(gaussianMask)) * np.asarray(
        L2)).tolist()
    blended_image = laplacian_to_image(Lc, L1FilterVec, coeff=np.ones(len(Lc)))
    return np.clip(blended_image, a_min=0, a_max=1)  # clip result to range [0,1]


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    image_1 = read_image(relpath("externals/hell.png"), 2)
    image_2 = read_image(relpath("externals/heaven.png"), 2)
    mask = (read_image(relpath("externals/mask1.png"), 1).round()).astype(bool)
    im_blend = np.zeros(image_1.shape)
    for i in range(3):  # running on rgb
        im_blend[:, :, i] = (pyramid_blending(image_1[:, :, i], image_2[:, :, i], mask, 10, 3, 3))

    fig, axs = plt.subplots(2, 2)  # i preferd to use subplots(2,2) instead of sublots(4) because it looks better
    fig.suptitle('Stairways to hell')
    axs[0][0].imshow(image_1)
    axs[0][1].imshow(image_2)
    axs[1][0].imshow(mask, cmap=GRAYCOLOR)
    axs[1][1].imshow(im_blend)
    plt.show()
    return image_1, image_2, mask, im_blend


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    image_1 = read_image(relpath("externals/star.png"), 2)
    image_2 = read_image(relpath("externals/golf.png"), 2)
    mask = (read_image(relpath("externals/mask2.png"), 1).round()).astype(bool)  # round and turn into boolean
    im_blend = np.zeros(image_1.shape)
    for i in range(3):  # running on rgb
        im_blend[:, :, i] = (pyramid_blending(image_1[:, :, i], image_2[:, :, i], mask, 5, 3, 3))
    fig, axs = plt.subplots(2, 2)  # i preferd to use subplots(2,2) instead of sublots(4) because it looks better
    fig.suptitle('Shooting Star')
    axs[0][0].imshow(image_1)
    axs[0][1].imshow(image_2)
    axs[1][0].imshow(mask, cmap=GRAYCOLOR)
    axs[1][1].imshow(im_blend)
    plt.show()
    return image_1, image_2, mask, im_blend


################################ Helpers ###############################
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


import os


def relpath(filename):
    """
    use relative paths
    :param filename: to image
    :return: path
    """
    return os.path.join(os.path.dirname(__file__), filename)


################################ ex1 functions ########################
#
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
