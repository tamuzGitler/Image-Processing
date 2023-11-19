################################ Imports ########################################
import imageio
import skimage.color

####### ex2_helper.py imports #######

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage.interpolation import map_coordinates

################################ Constants ######################################

GRAYCOLOR = "gray"
MAX_GREY_VALUE = 255
GRAYSCALE = 1
WAV_FILENAME = "ex2_presubmit\external\\aria_4kHz.wav"
CHANGE_RATE_FILENAME = "change_rate.wav"  # for private testing
CHANGE_SAMPLES_FILENAME = "change_samples.wav"  # for private testing


################################ Main functions #################################
####### task 1.1 #######

def DFT(signal):
    """
    transform a 1D discrete signal to its Fourier representation
    :param signal: n array of dtype float64 with shape (N,) or (N,1)
    :return: complex Fourier signal with the same shape
    """
    N = signal.shape[0]
    x = np.arange(N)  # range 0,N-1
    u = x.reshape((N, 1)) #vector of vectors
    expo = np.exp((-2j * np.pi * u * x) / N)
    fourier_signal = (signal.T @ expo).T  # -2j is imagenery=i
    return fourier_signal.astype(complex)


def IDFT(fourier_signal):
    """
    transform a 1D discrete signal to its Fourier representation
    :param fourier_signal: complex Fourier signal
    :return: n array of dtype float64 with shape (N,) or (N,1)
    """

    N = fourier_signal.shape[0]
    x = np.arange(N)  # range 0,N-1
    u = x.reshape(N, 1)  #vector of vectors
    expo = np.exp((2j * np.pi * u * x) / N)
    signal = (fourier_signal.T @ (expo)).T / N  # 2j is imagenery=i
    return np.real_if_close(signal).astype(complex)  # if imaginary parts are close to zero, return real parts


####### task 1.2 #######
def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: grayscale image of dtype float64
    :return: fourier_image
    """
    Fxv = np.apply_along_axis(func1d=DFT, axis=0, arr=image)  # compute fourier for current row
    Fuv = np.apply_along_axis(func1d=DFT, axis=1, arr=Fxv)  # compute fourier for current col
    return Fuv


def IDFT2(fourier_image):
    """
    convert a 2D Fourier representation to its discrete signal
    :param fourier_image: 2D array of dtype complex128
    :return: image
    """
    fx_V = np.apply_along_axis(func1d=IDFT, axis=0, arr=fourier_image)  # row
    fxy = np.apply_along_axis(func1d=IDFT, axis=1, arr=fx_V)  # col
    return fxy  # no need to divide by M*N because it happens in IDFT!!!


####### task 2.1 #######
def change_rate(filename, ratio):
    """
    Changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header
    :param filename:  representing the path to a WAV file
    :param ratio: positive float64 representing the duration change
    """
    sampleRate, data = wavfile.read(filename)
    new_rate = int(sampleRate * ratio)
    wavfile.write(CHANGE_RATE_FILENAME, new_rate, data)


####### task 2.2 #######
def change_samples(filename, ratio):
    """
    changes the duration of an audio file by reducing the number of samples using Fourier
    :param filename:  representing the path to a WAV file
    :param ratio: positive float64 representing the duration change
    """
    sampleRate, data = wavfile.read(filename)
    newSignal = resize(data, ratio)
    wavfile.write(CHANGE_SAMPLES_FILENAME, sampleRate, newSignal)
    return newSignal


def resize(data, ratio):
    """
    Changes the duration of an audio file by reducing the number of samples using Fourier
    :param data: 1D ndarray of dtype float64 or complex128 representing the original sample points
    :param ratio: positive float64 representing the duration change
    :return: a 1D ndarray of the dtype of data representing the new sample points.
    """
    dataType = data.dtype  # saves it for later :)
    fourier_signal = DFT(data)
    shifted_fourier_signal = np.fft.fftshift(
        fourier_signal)  # shift zero-frequency component to the center of the spectrum
    N = shifted_fourier_signal.size
    if ratio > 1:  # slow down
        slicing_factor = N - int(N / ratio)
        start = int(slicing_factor / 2)  # round down
        end = int(N - np.ceil(slicing_factor / 2))  # ceil -> round up
        fourier = shifted_fourier_signal[start:end]
    elif ratio < 1:  # make faster
        add = int((1 / ratio - 1) * shifted_fourier_signal.size)
        # padding
        if add % 2 != 0:  # odd
            left = np.zeros(add // 2, complex)
            right = np.zeros((add // 2) + 1, complex)
        else:  # even
            left = np.zeros(add // 2, complex)
            right = np.zeros((add // 2), complex)
        fourier = np.concatenate((left, shifted_fourier_signal, right), axis=None)
    elif ratio == 1:
        return data
    re_shift_fourier = np.fft.ifftshift(fourier)
    signal = IDFT(re_shift_fourier).astype(dataType)
    return signal


####### task 2.3 #######
def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling
    :param data:  1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same datatype as data.
    """
    dataType = data.dtype  # saves it for later :)
    spectrogram = stft(data)
    resized_spectogram = []
    for row in range(spectrogram.shape[0]):
        resized_spectogram.append(resize(spectrogram[row], ratio))
    resized_spectogram = np.asarray(resized_spectogram)
    speeded_data = istft(resized_spectogram).astype(dataType)
    return speeded_data


####### task 2.4 #######
def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram The difference from resize_spectrogram:
    includes the correction of the phases of each frequency according to the shift of each window
    :param data: 1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return: given data rescaled according to ratio with the same datatype as data
    """
    dataType = data.dtype  # saves it for later :)
    spectrogram = stft(data)
    speeded_data = phase_vocoder(spectrogram, ratio)
    speeded_data = istft(speeded_data)
    return speeded_data.astype(dataType)


####### task 3.1 #######
def conv_der(im):
    """
    computes the magnitude of image derivatives - using [0.5,0,-0.5]
    :param im: grayscale images of type float64
    :return: magnitude of the derivative, with the same dtype and shape
    """
    dataType = im.dtype  # saves it for later :)
    conv = np.array([0.5, 0, -0.5])
    x_conv = conv.reshape([1, 3])
    y_conv = conv.reshape([3, 1])
    dx = signal.convolve2d(im, x_conv, mode="same")  # Horizontal Derivative - x
    dy = signal.convolve2d(im, y_conv, mode="same")  # Vertical Derivative - y
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2).astype(dataType)
    return magnitude


####### task 3.2 #######
def fourier_der(im):
    """
    computes the magnitude of the image derivatives using Fourier transform
    :param im: grayscale images of type float64
    :return: magnitude of the derivative, with the same dtype and shape
    """
    dataType = im.dtype  # saves it for later :)
    fourier = DFT2(im)
    shifted_fourier = np.fft.fftshift(fourier)  # Shift (0,0) to center
    N, M = shifted_fourier.shape  # row,col
    coefficient = (2j * np.pi)

    # Horizontal Derivative - x
    row_start, row_end = int(-N / 2), np.ceil(N / 2)
    u = np.arange(row_start, row_end)
    dx = ((coefficient / N )* (shifted_fourier.T * u)).T  # tranform fourier for multiplication and then reverse
    dx = IDFT2(np.fft.ifftshift(dx))

    # Vertical Derivative - y
    col_start, col_end = int(-M / 2), np.ceil(M / 2)
    v = np.arange(col_start, col_end)
    dy = (coefficient / M) * shifted_fourier * v
    dy = IDFT2(np.fft.ifftshift(dy)) #shift back and compute idft2

    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2).astype(dataType)

    return magnitude


################################ ex2_helper.py #####################################


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


################################ from sol1.py #################################

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

