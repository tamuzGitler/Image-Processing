################################ Imports ######################################

import imageio
import numpy as np
import skimage.color
import matplotlib.pyplot as plt

################################ Constants ######################################

GRAYCOLOR = "gray"
MAX_GREY_VALUE = 255
GRAYSCALE = 1
RGB = 2
RGB_DIMMENSION = 3
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])
INVERSE_RGB_YIQ_TRANSFORMATION_MATRIX = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)
BINS = 256


# FILENAME = "ex1_presubmit\presubmit_externals\jerusalem.jpg" #for private testing
# FILENAME2 = "ex1_presubmit\presubmit_externals\monkey_black_and_white.jpeg" #for private testing

################################ Main functions ######################################

def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imageio.imread(filename)  # to run write imageio.v2.imread

    if (representation == GRAYSCALE):
        image = skimage.color.rgb2gray(image)
    else:  # RGB CASE
        image = image / MAX_GREY_VALUE  # normalize

    return image


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    image = read_image(filename, representation)

    if (representation == GRAYSCALE):
        plt.imshow(image, cmap=GRAYCOLOR, vmin=0, vmax=MAX_GREY_VALUE)  # show in grey
    elif (representation == RGB):
        plt.imshow(image)

    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """

    reshapedRGB = imRGB.reshape(imRGB.shape[0] * imRGB.shape[1], 3).T
    imYIQ = (RGB_YIQ_TRANSFORMATION_MATRIX @ reshapedRGB)
    imYIQ = imYIQ.T.reshape(imRGB.shape)
    return imYIQ


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """

    reshapedYIQ = imYIQ.reshape(imYIQ.shape[0] * imYIQ.shape[1], 3).T
    imRGB = (INVERSE_RGB_YIQ_TRANSFORMATION_MATRIX @ reshapedYIQ)
    imRGB = imRGB.T.reshape(imYIQ.shape)
    return imRGB


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    image = im_orig
    # check if im_orig is rgb presentation
    if (len(im_orig.shape) == RGB_DIMMENSION):
        imYIQ = rgb2yiq(im_orig)
        image = imYIQ[:, :, 0]

    # step 1-2
    flattern_img = (image.flatten() * MAX_GREY_VALUE).astype(
        int)  # because im_orig is in range [0,1] and we need indexs
    hist_orig, bins = np.histogram(image * MAX_GREY_VALUE, bins=BINS, range=[0, MAX_GREY_VALUE])
    cumulative_hist = np.cumsum(hist_orig)

    # step 3-6
    try:
        norm_cumulative_hist = normalize_cumulative_hist(cumulative_hist)
    except ZeroDivisionError:
        return [im_orig, hist_orig, hist_orig]  # cannot equalize so hist_orig=hist_eq & im_eq=im_orig
    if (len(im_orig.shape) == RGB_DIMMENSION):
        imYIQ[:, :, 0] = np.reshape(norm_cumulative_hist[flattern_img] / MAX_GREY_VALUE,
                                    [im_orig.shape[0], im_orig.shape[1]])  # divide by 255 to get Y values between [0,1]
        im_eq = yiq2rgb(imYIQ)
    else:
        im_eq = np.reshape(norm_cumulative_hist[flattern_img], im_orig.shape)  # step 7 - map the intensity values
        im_eq /= MAX_GREY_VALUE  # the equalized image. grayscale or RGB float64 image with values in [0, 1].
    hist_eq = np.histogram(im_eq * MAX_GREY_VALUE, bins=BINS, range=(0, 1))[
        0]  # is a 256 bin histogram of the equalized image (array with shape (256,) ).

    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    # check if im_orig is rgb presentation
    image = im_orig  # because image is in range [0,1]

    if (len(im_orig.shape) == RGB_DIMMENSION):
        imYIQ = rgb2yiq(im_orig)
        image = imYIQ[:, :, 0]

    # init values
    error = []  # array that counts the error
    Q = np.zeros(n_quant)  # intensities map
    hist_orig, bins = np.histogram(image, bins=256)
    cumulative_hist = np.cumsum(hist_orig)
    num_of_pixels = cumulative_hist[-1]
    H = hist_orig / num_of_pixels  # weights
    G = np.arange(0, 256)
    Z = init_z(cumulative_hist, n_quant, num_of_pixels)  # init z=borders according to n_quant
    update_q(G, H, Q, Z, n_quant)  # init Q

    # iterate between Z and Q, each iteration calcs the current error
    for iter in range(n_iter):
        prev_z = np.copy(Z)  # saves Z before update to check if it hasn't changed
        update_z(Q, Z, n_quant)
        update_q(G, H, Q, Z, n_quant)
        calc_error(G, H, Q, Z, error, n_quant)
        if np.allclose(prev_z, Z):
            break

    image = map_shades_to_q(Q, Z, image * MAX_GREY_VALUE, n_quant)
    # handels rgb
    if (len(im_orig.shape) == RGB_DIMMENSION):
        imYIQ[:, :, 0] = image
        im_quant = yiq2rgb(imYIQ)
    else:
        im_quant = image
    return im_quant, error


################################ Helpers ######################################

def normalize_cumulative_hist(cumulative_hist):
    """
    normalizes the cumulative hist by the form from lecture
    :param cumulative_hist:  cumulative histogram
    :return: normalized histgoram / throws exception when cant normalize
    """
    numerator = (cumulative_hist - cumulative_hist.min()) * MAX_GREY_VALUE
    denomenator = cumulative_hist.max() - cumulative_hist.min()
    try:
        norm_cumulative_hist = np.round(numerator / denomenator)  # T - from lecture
    except ZeroDivisionError:
        raise Exception("zero division!")
    return norm_cumulative_hist


def map_shades_to_q(Q, Z, image, n_quant):
    """
    maps image to contain n_quant shades - Q
    :param Q: shades
    :param Z: borders
    :param image: im_orig
    :param n_quant: Number of intensities im_quant image will have
    :return: image with n_quant shades - Q
    """
    look_up_table = np.zeros(256).astype(int)
    for i in range(n_quant):
        start, end = (np.round(Z[i] + 1)).astype(int), (np.round(Z[i + 1]) + 1).astype(int)
        range_z = np.arange(start, end)
        look_up_table[range_z] = Q[i]
    image = (image).astype(
        int)  # because image is in range [0,1] and we want to map we need range [0,255]
    image = look_up_table[image] / MAX_GREY_VALUE  # return to [0,1] range
    return image


def calc_error(G, H, Q, Z, error, n_quant):
    """
    :param H:histogram distribution
    :param Q:shades
    :param Z:borders
    :param error: error array
    :param n_quant: Number of intensities im_quant image will have
    """
    total_error = 0
    for i in range(n_quant):
        start, end = Z[i].astype(int) + 1, Z[i + 1].astype(int) + 1
        g = G[start:end]
        h_g = H[g]
        cur_err = np.power(Q[i] - g, 2) * h_g
        total_error += cur_err.sum()
    error.append(total_error)


def update_q(G, H, Q, Z, n_quant):
    """
    updates Q value from form in lecture
    :param G: array between [0,255]
    :param H: histogram distribution
    :param Q: shades
    :param Z: borders
    :param n_quant:  Number of intensities im_quant image will have
    :return: updated q
    """
    for i in range(n_quant):
        start, end = Z[i].astype(int) + 1, Z[i + 1].astype(int) + 1
        H_g = H[start:end]  # weights in range of g!
        g = G[start:end]
        Q[i] = np.sum(g * H_g) / np.sum(H_g)


def update_z(Q, Z, n_quant):
    """
    updates z value from form in lecture
    :param Q: shades
    :param Z: borders
    :param n_quant:  Number of intensities im_quant image will have
    :return: updated z
    """
    for k in range(1, n_quant):
        Z[k] = (Q[k - 1] + Q[k]) / 2


def init_z(cumulative_hist, n_quant, num_of_pixels):
    """
    smart inition of Z
    :param cumulative_hist:  cumulative histogram
    :param n_quant: Number of intensities im_quant image will have
    :param num_of_pixels: number of pixels in image
    :return: Z
    """
    average_pixel_intensity = (num_of_pixels // n_quant)  # the average pixels per quant
    cumulative_diff = (cumulative_hist // average_pixel_intensity)  # for finding where the borders of the histogram
    Z = np.where(np.diff(cumulative_diff, prepend=np.nan))[0]  # index's that divide the histograms into segments
    Z[0] = -1  # plaster :)
    return Z.astype(float)


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    pass
