import numpy as np
import cv2
from ex1_utils import gaussderiv, gausssmooth

def horn_schunck(im1, im2, n_iters, conv=False, lmbd=0.5, sigma1=1.0, u_init=None, v_init=None, derivative_smoothing=False, sigma2=1.0):
    """
    Compute the Horn-Schunck optical flow estimation.
    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        im1 (np.ndarray): First frame.
        im2 (np.ndarray): First frame.
        n_iters (int): Number of iterations to perform.
        conv (bool): Flag specifying whether to use convergence criterion to stop iteration or not.
        sigma1 (float): Standard deviation of the Gaussian smoothing filter used in computing
        the spatial derivatives.
        u_init (numpy.ndarray): Initialization for the u array.
        v_init (numpy.ndarray): Initialization for the v array.
        derivative_smoothing (bool): Boolean flag indicating whether to use smoothing of spatial and the
        temporal derivative.
        sigma2 (float): Standard deviation of the Gaussian smoothing filter used in smoothing
        the temporal derivative.

    Returns:
        (tuple): tuple of matrices of changes in spatial
        coordinates for each pixel.

    """

    # If initial values of u and v are given.
    if not u_init is None and not v_init is None:
        u = u_init
        v = v_init
    else:
        # Else initialize with zeros.
        u = np.zeros(im1.shape, dtype=im1.dtype)
        v = np.zeros(im1.shape, dtype=im1.dtype)

    # Smooth and compute derivatives of first image.
    im_df_dx_t1, im_df_dy_t1 = gaussderiv(im1, sigma=sigma1)

    # Compute temporal derivative
    im_df_dt = im2 - im1
    
    # If option to smooth derivatives selected, average the spatial
    # derivatives and smooth the temporal derivative using a gaussian.
    if derivative_smoothing:
        im_df_dx_t2, im_df_dy_t2 = gaussderiv(im2, sigma=sigma1)
        im_df_dx = 0.5*(im_df_dx_t1 + im_df_dx_t2)
        im_df_dy = 0.5*(im_df_dy_t1 + im_df_dy_t2)
        im_df_dt = gausssmooth(im2, sigma2) - gausssmooth(im1, sigma2)
    else:
        im_df_dx = im_df_dx_t1
        im_df_dy = im_df_dy_t1
    
    # Define residual Laplacian kernel.
    res_lap = np.array([[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]])
    
    # u and v matrices for previous iteration (used to check for convergence).
    u_prev = u
    v_prev = v

    # Convergence criterion.
    CONV_THRESH = 1.0e-1

    # Iteratively perform Horn-Schunck method.
    for iter_count in np.arange(n_iters):

        # Convolve u and v values with residual Laplacian kernel.
        u_a = cv2.filter2D(u, ddepth=-1, kernel=res_lap)
        v_a = cv2.filter2D(v, ddepth=-1, kernel=res_lap)
        
        # Compute the new values of u and v.
        p = (im_df_dx*u_a + im_df_dy*v_a + im_df_dt)/(im_df_dx**2 + im_df_dy**2 + lmbd)
        u = u_a - im_df_dx*p
        v = v_a - im_df_dy*p
        
        # If using convergence criterion, check for convergence.
        if conv:
            if np.trace(np.matmul(u - u_prev, (u - u_prev).T)) < CONV_THRESH and np.trace(np.matmul(v - v_prev, (v - v_prev).T)) < CONV_THRESH:
                break

        # Set new previous u and v values.
        u_prev = u
        v_prev = v


    # Return approximated changes in spatial coordinates as
    # a tuple of numpy arrays (matrices).
    return -u, -v

