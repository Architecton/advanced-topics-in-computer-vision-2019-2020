import numpy as np
import cv2

from gaussian_pyramid import gaussian_pyramid
from lucas_kanade import lucas_kanade


def warp_image(img, u, v):
    """
    Warp image using computed optical flow.

    Args:
        img (numpy.ndarray): Image to warp.
        u (numpy.ndarray): Computed optical flow values for the x direction.
        v (numpy.ndarray): Computed optical flow values for the y direction.

    Returns:
       (numpy.ndarray): Warped image. 
    """

    # Allocate matrix for warped image.
    res = np.zeros(img.shape, dtype=img.dtype)

    # Go over pixels in original image and map to new locations.
    for idx1 in np.arange(img.shape[0]):
        for idx2 in np.arange(img.shape[1]):
            res[min(max(idx1 + int(round(v[idx1, idx2])), 0), res.shape[0]-1), min(max(idx2 + int(round(u[idx1, idx2])), 0), res.shape[1]-1)] = img[idx1, idx2]
    
    return res


def iterative_lucas_kanade(im1, im2, n):
    """
    Compute iterative Lucas-Kanade algorithm for specified frames.
    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        im1 (np.ndarray): First frame.
        im2 (np.ndarray): First frame.
        n (int): Size of neighborhood to use in computations.
    Returns:
        (tuple): tuple of matrices of changes in spatial
        coordinates for each pixel.
    """

    # Compute gaussian pyramids from the two frames.
    gaussian_pyramid_im1 = gaussian_pyramid(im1)
    gaussian_pyramid_im2 = gaussian_pyramid(im2)

    # Set offset for pyramid levels (how many levels at top to skip).
    START_OFFSET = 0

    # Set flag indicating first level.
    first_flag = True

    # Go over levels in pyramid and iteratively compute Lucas-Kanade algorithm.
    for idx in np.arange(START_OFFSET, len(gaussian_pyramid_im1)):
        
        # Get images on current level of Gaussian pyramid.
        im1_nxt = gaussian_pyramid_im1[idx]
        im2_nxt = gaussian_pyramid_im2[idx]
        
        # If on first level (top-most)...
        if first_flag:

            # Compute optical flow.
            u, v = lucas_kanade(im1_nxt, im2_nxt, n=n)
            
            # Upsample flow to match image at next lower level.
            dim_match = gaussian_pyramid_im1[idx+1].shape
            u_upsampled = cv2.resize(u, dim_match)
            v_upsampled = cv2.resize(v, dim_match)

            # Set flag indicating first level to false.
            first_flag = False
        else:
            # Warp image using upsampled from from previous level.
            im2_nxt_warped = warp_image(im2_nxt, u_upsampled, v_upsampled)

            # Estimate residual flow using Lucas-Kanade method.
            u_res, v_res = lucas_kanade(im1_nxt, im2_nxt_warped, n=n)

            # Add upsampled flow and residual flow to get flow on current level.
            u_fin = u_upsampled + u_res
            v_fin = v_upsampled + v_res
            
            # If not at bottom-most level (last level)...
            if idx < len(gaussian_pyramid_im1) - 1:
                dim_match = gaussian_pyramid_im1[idx+1].shape
                u_upsampled = cv2.resize(u_fin, dim_match)
                v_upsampled = cv2.resize(v_fin, dim_match)
    
    # Return flow computed on bottom-most level.
    return u_fin, v_fin

