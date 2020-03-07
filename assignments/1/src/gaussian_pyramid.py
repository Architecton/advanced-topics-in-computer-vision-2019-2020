import numpy as np
from ex1_utils import gausssmooth

def gaussian_pyramid(im):
    """
    Compte Gaussian pyramid for specified image and return levels as list of numpy arrays (matrices).
    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        im (numpy.ndarray): Image for which to compute the Gaussian pyramid.

    Returns:
        (list): List of numpy arrays (matrices) representing the Gaussian level pyramids.
    """
    
    # Set sigma for Gaussian smoothing.
    SIGMA = 1.0

    # Ensure image image dimensions are even integers.
    im_formatted = im[:im.shape[0]-np.mod(im.shape[0], 2), :im.shape[1]-np.mod(im.shape[1], 2)]

    # Initialize results list.
    res = []

    # Add original image to results array.
    res.append(im_formatted)

    # While image represented by matrix whose lowest dimension larger than 2 ...
    im_nxt = im_formatted.copy()
    while np.min(im_nxt.shape) > 1:

        # Smooth image using Gaussian kernel and subsample.
        im_nxt = gausssmooth(im_nxt[:im_nxt.shape[0]-np.mod(im_nxt.shape[0], 2), :im_nxt.shape[1]-np.mod(im_nxt.shape[1], 2)], sigma=SIGMA)[:-1:2, :-1:2]

        # Add subsampled image to results list.
        res.append(im_nxt.copy())
    
    # Return list of Gaussian pyramid stages.
    return list(reversed(res))


### Test ###
if __name__ == '__main__':
    from PIL import Image
    image = Image.open('../disparity/cporta_left.png')
    data = np.asarray(image)
    res = gaussian_pyramid(data)

