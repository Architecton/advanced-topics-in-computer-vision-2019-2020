import math

import numpy as np
from scipy import signal
import cv2


def gausssmooth(img, sigma):
    # Create Gaussian kernel and filter image with it.
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    return cv2.sepFilter2D(img, -1, G, G)


def generate_responses_1():
    # Generate responses map.
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    return gausssmooth(responses, 10)


def generate_responses_2():
    # Generate responses map.
    responses = np.zeros((1000, 1000), dtype=np.float32)
    val = 1/800
    incr = 1/800
    gauss_sig = signal.gaussian(1000, std=400)
    for idx in np.arange(800):
        responses[idx, :] = val*gauss_sig
        val += incr
    return gausssmooth(responses, 40)


def get_patch(img, center, sz):
    
    # Crop coordinates.
    x0 = round(int(center[0] - sz[0] / 2))
    y0 = round(int(center[1] - sz[1] / 2))
    x1 = int(round(x0 + sz[0]))
    y1 = int(round(y0 + sz[1]))
    
    # Set padding - how far across the image border is the edge of patch?
    x0_pad = max(0, -x0)
    x1_pad = max(x1 - img.shape[1] + 1, 0)
    y0_pad = max(0, -y0)
    y1_pad = max(y1 - img.shape[0] + 1, 0)

    # Crop target.
    if len(img.shape) > 2:
        # BGR image.
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad, :]
    else:
        # Grayscale image.
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]
    
    # Cropped and padded image.
    im_crop_padded = cv2.copyMakeBorder(img_crop, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_REPLICATE)

    # Crop mask tells which pixels are within the image (1) and which are outside (0).
    m_ = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    crop_mask = m_[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]
    crop_mask = cv2.copyMakeBorder(crop_mask, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_CONSTANT, value=0)

    # Return cropped and padded image and crop mask.
    return im_crop_padded, crop_mask


def create_epanechnik_kernel(width, height, sigma):

    # Get floored halves of width and height (should be odd).
    w2 = int(math.floor(width / 2))
    h2 = int(math.floor(height / 2))
    
    # Create meshgrid and normalize values to interval [0, 1].
    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    X = X / np.max(X)
    Y = Y / np.max(Y)
    
    # Create kernel.
    kernel = (1 - ((X / sigma)**2 + (Y / sigma)**2))

    # Normalize kernel to interval [0, 1] and
    # clip negative values.
    kernel = kernel / np.max(kernel)
    kernel[kernel < 0] = 0

    # Return kernel.
    return kernel


def extract_histogram(patch, nbins, weights=None):

    # Note: input patch must be a BGR image (3 channel numpy array).
    # Convert each pixel intensity to the one of nbins bins.
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    
    # Calculate bin index of a 3D histogram.
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # Count bin indices to create histogram (use per-pixel weights if given).
    if weights is not None:
        histogram_ = np.bincount(bin_idxs.flatten(), weights=weights.flatten())
    else:
        histogram_ = np.bincount(bin_idxs.flatten())
    
    # zero-pad histogram (needed since bincount function does not generate histogram with nbins**3 elements).
    histogram = np.zeros((nbins**3, 1), dtype=histogram_.dtype).flatten()
    histogram[:histogram_.size] = histogram_

    # Return computed histogram.
    return histogram


def backproject_histogram(patch, histogram, nbins):
    
    # Note: input patch must be a BGR image (3 channel numpy array).
    # Convert each pixel intensity to the one of nbins bins.
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
   
    # Calculate bin index of a 3D histogram.
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # Use histogram as a lookup table for pixel backprojection.
    backprojection = np.reshape(histogram[bin_idxs.flatten()], (patch.shape[0], patch.shape[1]))

    # Return computed backprojection.
    return backprojection


# Base class for tracker.
class Tracker():
    def __init__(self, params):
        self.parameters = params

    def initialize(self, image, region):
        raise NotImplementedError

    def track(self, image):
        raise NotImplementedError

