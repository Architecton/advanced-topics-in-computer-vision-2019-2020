import numpy as np
from ex2_utils import Tracker, get_patch
from ex3_utils import create_gauss_peak, gauss_c, apply_window
import cv2

class MosseTracker(Tracker):
    """
    Implementation of the simplified MOSSE tracking algorithm.
    Author: Jernej Vivod

    Args:
        params (obj): MosseParams instance specifying the parameters for the tracker.
    """
    
    def initialize(self, img, region):

        # Set tracker name.
        self.name = "mosse-tracker"

        # Get initialized position.
        self.pos = [region[0] + region[2]/2, region[1] + region[3]/2]

        # Set tracking patch size.
        self.size = (int(region[2]) + abs(int(region[2]) % 2 - 1), 
                      int(region[3]) + abs(int(region[3]) % 2 - 1))
        
        # Get initial filter.
        xx, yy = np.meshgrid(np.arange(1, img.shape[1]+1), np.arange(1, img.shape[0]+1))
        gauss_im = gauss_c(xx, yy, sig=100, center=self.pos)
        self.gauss_fd = np.fft.fft2(get_patch(gauss_im, self.pos, self.size)[0])
        feature_patch = apply_window(cv2.cvtColor(get_patch(img, self.pos, self.size)[0], cv2.COLOR_BGR2GRAY))
        feature_patch_fd = np.fft.fft2(feature_patch)
        self.filt = (self.gauss_fd * np.conj(feature_patch_fd))/(feature_patch_fd * np.conj(feature_patch_fd) + self.parameters.lmbd)
        
   

    def track(self, img):

        feature_patch = cv2.cvtColor(get_patch(img, self.pos, self.size)[0], cv2.COLOR_BGR2GRAY)
        feature_patch_proc = apply_window(feature_patch)
        feature_patch_fd = np.fft.fft2(feature_patch_proc)
        loc = np.fft.ifft2(self.filt*feature_patch_fd)
        

        new_pos = np.where(np.abs(loc) == np.max(np.abs(loc)))
        y = np.mean(new_pos[0])
        x = np.mean(new_pos[1])


        dy = y - self.size[1]/2 + 1
        dx = x - self.size[0]/2 + 1

        self.pos[0] += dx 
        self.pos[1] += dy

        
        return [self.pos[0] - self.size[0]//2+1, self.pos[1] - self.size[1]//2+1, self.size[0], self.size[1]]



class MosseParams():
    """
    Encapsulation of the MOSSE tracking algorithm parameters.

    Args:
        lmbd (float): the lambda parameter weighting the influence
        of the kernel size in the cost function.
    """

    def __init__(self, lmbd, alpha):
        self.lmbd = lmbd
        self.alpha = alpha


