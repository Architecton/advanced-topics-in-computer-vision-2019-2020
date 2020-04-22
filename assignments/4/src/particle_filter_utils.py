import numpy as np
import random
import math
import cv2


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

def calc_histogram(patch, num_bins=50):
    """ Color histogram is calculated and the 3 histograms
    are appended into a one-dimensional vector and normalized. """

    # used to calculate histogram for a part of the image
    # but extract_patch() is used instead of this.
    mask = None

    blue_model = cv2.calcHist([patch], [0], mask, [num_bins],  [0,256]).flatten()
    green_model = cv2.calcHist([patch], [1], mask, [num_bins],  [0,256]).flatten()
    red_model = cv2.calcHist([patch], [2], mask, [num_bins],  [0,256]).flatten()

    color_patch = np.concatenate((blue_model, green_model, red_model))
    
    if np.sum(color_patch) == 0:
        return np.zeros(len(color_patch))

    # Normalize histogram values for the KL divergence computation
    color_patch = color_patch/np.sum(color_patch)
    return color_patch



def evaluate_likelihood(image, particles, obs_func, ref):
    """
    Evaluate p(y_{k}|x_{k}) for each particle in specified tuple of particles and their weights.

    Args:
        image (numpy.ndarray): Image on which the particles are evaluated.
        particles (tuple): Particles arranged as a matrix and their weights as a vector.
        obs_func (function): Evaluation function that tkes the state, image and reference model
        and returns new weight for the particle.
        ref (numpy.ndarray): Reference model for comparison.

    Returns: 
        (tuple): Particles with updated weights.
    """

    # Evaluate each particle and compute new weights.
    for idx, state in enumerate(particles[0]):
        particles[1][idx] = obs_func(image, state, ref)

    # Normalize weights.
    particles[1] /= np.sum(particles[1])

    # Return particles with updated weights.
    return particles


def discrete_kl_divergence(P, Q):
    """ Calculates the Kullback-Lieber divergence
    according to the discrete definition:
    sum [P(i)*log[P(i)/Q(i)]]
    where P(i) and Q(i) are discrete probability
    distributions. In this case the one """

    """ Epsilon is used here to avoid conditional code for
    checking that neither P or Q is equal to 0. """
    epsilon = 0.00001

    # To avoid changing the color model, a copy is made
    temp_P = P+epsilon
    temp_Q = Q+epsilon

    divergence=np.sum(temp_P*np.log(temp_P/temp_Q))
    return divergence

def get_new_ref(img, pos, model):
    # Extract reference patch and convert to HSV color space.
    patch_ref = img[int(round(max(pos[1], 0))):int(round(min(pos[1]+pos[3]+1, img.shape[0]))), 
                    int(round(max(pos[0], 0))):int(round(min(pos[0]+pos[2]+1, img.shape[1]))), :]

    # kern = create_epanechnik_kernel(pos[2], pos[3], 2)
    return calc_histogram(patch_ref)


def get_ref(img, pos, model, compare_method=None):
    if model == 'hist-hsv':
        
        # Set number of bins.
        H_BINS = 50
        S_BINS = 50
        
        # Extract reference patch and convert to HSV color space.
        patch_ref = img[int(round(max(pos[1], 0))):int(round(min(pos[1]+pos[3]+1, img.shape[0]))), 
                        int(round(max(pos[0], 0))):int(round(min(pos[0]+pos[2]+1, img.shape[1]))), :]

        # kern = create_epanechnik_kernel(pos[2], pos[3], 2)
        ref = calc_histogram(patch_ref)

        # Crmpute and normalize reference histogram.
        # patch_ref_hsv = patch_ref # cv2.cvtColor(patch_ref, cv2.COLOR_BGR2HSV)
        # ref = cv2.calcHist([patch_ref_hsv], [0, 1, 2], None, [H_BINS, S_BINS, H_BINS], ranges=[0, 256, 0, 256, 0, 256], accumulate=False)
        # cv2.normalize(ref, ref, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Define function for comparing state with reference.
        def res(img, state, hist_ref):

            try:
            
                # Extract patch from image and convert to HSV color space.
                patch = img[int(round(max(state[1], 0))):int(round(min(state[1]+state[3]+1, img.shape[0]))), 
                            int(round(max(state[0], 0))):int(round(min(state[0]+state[2]+1, img.shape[1]))), :]

                hist_p = calc_histogram(patch)
                # patch_hsv = patch # cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

                # # Compute and normalize histogram.
                # hist_p = cv2.calcHist([patch_hsv], [0, 1, 2], None, [H_BINS, S_BINS, H_BINS], ranges=[0, 180, 0, 256, 0, 256], accumulate=False)
                # cv2.normalize(hist_p, hist_p, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # hist_p = extract_histogram(patch, )

                # l = 1 # lambda
                # divergence = discrete_kl_divergence(hist_ref, hist_p)
                # likelihood = np.exp(-l*divergence)
                # return likelihood
                

                dist = np.sqrt(1 - np.sum(np.sqrt(hist_p * hist_ref)))

                # Compare histograms.
                sol = 1/(np.sqrt(2*np.pi*0.2))*np.exp(-((dist**2)/(2*0.2**2)))
                return sol
            except:
                return 0.0
        
        
        return ref, res

    if model == 'hist-rgb':
        # TODO
        pass
    if model == 'hog':
        # TODO
        pass

