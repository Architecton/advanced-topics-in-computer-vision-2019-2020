import numpy as np
from ex1_utils import gaussderiv
import cv2

def lucas_kanade(im1, im2, n):
    """
    Compute the Lucas-Kanade optical flow estimation.
    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        im1 (np.ndarray): First frame.
        im2 (np.ndarray): First frame.
        n (int): Size of neighborhood to use in computations.

    Returns:
        (tuple): tuple of matrices of changes in spatial
        coordinates for each pixel.
    """

    # Smooth and compute derivatives of first image.
    im_df_dx, im_df_dy = gaussderiv(im2, sigma=1.0)

    # Compute temporal derivative
    im_df_dt = im2 - im1
    
    # Initialize kernel for computing the neighborhood sums.
    sum_ker = np.ones((n, n), dtype=float)

    # Compute required neighborhood sums.
    sum_x_squared = cv2.filter2D(np.power(im_df_dx, 2), ddepth=-1, kernel=sum_ker) #
    sum_y_squared = cv2.filter2D(np.power(im_df_dy, 2), ddepth=-1, kernel=sum_ker) #
    sum_xy = cv2.filter2D(im_df_dx*im_df_dy, ddepth=-1, kernel=sum_ker)
    sum_xt = cv2.filter2D(im_df_dx*im_df_dt, ddepth=-1, kernel=sum_ker)
    sum_yt = cv2.filter2D(im_df_dy*im_df_dt, ddepth=-1, kernel=sum_ker)
    
    # Compute determinant of the A matrix.
    det = sum_x_squared*sum_y_squared - sum_xy*sum_xy

    # Set minimum determinant value and get indices of pixels where
    # determinant above set threshold.
    # DET_THRESH = 0.1
    # det_idx = np.abs(det) > DET_THRESH
    
    # Compute the approximated changes in spatial coordinates.

    # u = np.zeros(im1.shape, dtype=float)
    # v = np.zeros(im2.shape, dtype=float)
    # u[det_idx] = -(sum_y_squared * sum_xt - sum_xy * sum_yt)[det_idx]/(det[det_idx])
    # v[det_idx] = -(sum_x_squared * sum_yt - sum_xy * sum_xt)[det_idx]/(det[det_idx])

    u = -(sum_y_squared * sum_xt - sum_xy * sum_yt)/(det)
    v = -(sum_x_squared * sum_yt - sum_xy * sum_xt)/(det)
    
    # Return approximated changes in spatial coordinates as
    # a tuple of numpy arrays (matrices).
    return -u, -v

