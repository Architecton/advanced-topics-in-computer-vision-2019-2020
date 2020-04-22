import numpy as np
import cv2
import os
import time
from utils.tracker import Tracker
# from ex2_utils import Tracker


class MosseParams:
    """
    Encapsulation of the MOSSE tracking algorithm parameters.

    Args:
        lmbd (float): The lambda parameter used to prevent zero-division errors when constructing
        correlation filter.
        alpha (float): Alpha value weighting the previous filter when updating it in the next iteration.
        training_iter (int): Number of additional training iterations on object of interest
        to perform.
        rotate (bool): Rotate the object of interest when performing additional training iterations or not.
    """

    def __init__(self, lmbd=0.1, alpha=0.01, sig=20, training_iter=10, rotate=False, scale=1.0, measure_runtime=False):
        self.lmbd = lmbd
        self.alpha = alpha
        self.sig = sig
        self.training_iter = training_iter
        self.rotate = rotate
        self.scale = scale
        self.measure_runtime = measure_runtime


class MosseTracker(Tracker):
    """
    Implementation of the MOSSE tracking algorithm.
    Author: Jernej Vivod

    Args:
        parameters (obj): MosseParams instance specifying the parameters for the tracker.
    """

    def __init__(self, parameters=MosseParams()):
        self.parameters = parameters
        if self.parameters.measure_runtime:
            self.init_count = 0
            self.track_count = 0
            self.total_init_time = 0
            self.total_track_time = 0


    def name(self):
        return "Mosse Tracker"


    def initialize(self, img, region):
        """
        Initialize Mosse tracker.

        Args:
            img (numpy.ndarray): First image.
            region (list): bounding box specification for the
            object on the first image. First and second values
            represent the position of the left-upper corner. The
            third and fourth values represent the width and the 
            height of the bounding box respectively.

        """

        if self.parameters.measure_runtime:
            self.init_count += 1
            time_start = time.time()
            
        
        # get initial image and ground truth (bounding box)
        img_init = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.init_gt = np.array(region).astype(int)
        
        # Set initial position and initial inbound position.
        self.pos = self.init_gt.copy()
        self.inb_pos = np.array([self.pos[0], self.pos[1], self.pos[0] + self.pos[2], self.pos[1] + self.pos[3]]).astype(np.int64)
        
        # Get ideal filter response on original image.
        gauss_im = self._gauss_image(img_init, self.init_gt, self.parameters.sig)

        # Extract region of interest from ideal filter response image and convert to frequency domain.
        gauss_ext = gauss_im[self.init_gt[1]:self.init_gt[1]+self.init_gt[3], self.init_gt[0]:self.init_gt[0]+self.init_gt[2]]
        self.gauss_fd = np.fft.fft2(gauss_ext)
        
        # Apply filter area training patch scaling and extract reagion of interest.
        change_ub = int(round((self.init_gt[3] * self.parameters.scale - self.init_gt[3])/2))
        change_lr = int(round((self.init_gt[2] * self.parameters.scale - self.init_gt[2])/2))
        u_bound = max(self.init_gt[1] - change_ub, 0)
        b_bound = min(self.init_gt[1] + self.init_gt[3] + change_ub, img_init.shape[0])
        l_bound = max(self.init_gt[0] - change_lr, 0)
        r_bound = min(self.init_gt[0] + self.init_gt[2] + change_lr, img_init.shape[1])
        img_ext = cv2.resize(img_init[u_bound:b_bound, l_bound:r_bound], (self.init_gt[2], self.init_gt[3]))
        
        # Construct filter numerator and denominator.
        self.filt_num, self.filt_denom = self._construct_filter(img_ext, self.gauss_fd)
        # self.filt_num *= self.parameters.alpha
        # self.filt_denom *= self.parameters.alpha
        
        if self.parameters.measure_runtime:
            self.total_init_time += time.time() - time_start


    def track(self, img):
        """
        Perform tracking on next image using computed filter.

        Args:
            img (numpy.ndarray): Image on which to localize the object using
            the computed filter

        Returns:
            (list): bounding box specification for the
            object on the first image. First and second values
            represent the position of the left-upper corner. The
            third and fourth values represent the width and the
            height of the bounding box respectively.
        """

        if self.parameters.measure_runtime:
            self.track_count += 1
            time_start = time.time()

        # Convert next image frame to grayscale.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Construct filter.
        filt = self.filt_num/(self.filt_denom + self.parameters.lmbd)

        # Preprocess image - normalize and apply Hanning window.
        img_prep = self._preprocess(cv2.resize(img_gray[self.inb_pos[1]:self.inb_pos[3], 
            self.inb_pos[0]:self.inb_pos[2]], (self.init_gt[2], self.init_gt[3])))

        # Get filter response.
        loc = np.fft.ifft2(filt * np.fft.fft2(img_prep))
        
        # Find positions of maximum values in filter response.
        # Get changes in x and y coordinates.
        max_val = np.max(loc)
        pos_max = np.where(loc == max_val)
        delta_y = int(np.mean(pos_max[0]) - loc.shape[0] / 2)
        delta_x = int(np.mean(pos_max[1]) - loc.shape[1] / 2)
        
        # Add estimated changes in position to current position.
        self.pos[0] = self.pos[0] + delta_x
        self.pos[1] = self.pos[1] + delta_y
        
        # Get inbound position by clipping current position with respect to image borders.
        self.inb_pos[0] = np.clip(self.pos[0], 0, img.shape[1])
        self.inb_pos[1] = np.clip(self.pos[1], 0, img.shape[0])
        self.inb_pos[2] = np.clip(self.pos[0] + self.pos[2], 0, img.shape[1])
        self.inb_pos[3] = np.clip(self.pos[1] + self.pos[3], 0, img.shape[0])
        self.inb_pos = self.inb_pos.astype(np.int64)
        
        # Extract region of interest using new estimated position.
        img_mov = img_gray[self.inb_pos[1]:self.inb_pos[3], self.inb_pos[0]:self.inb_pos[2]]
        img_mov = self._preprocess(cv2.resize(img_mov, (self.init_gt[2], self.init_gt[3])))
        
        # Update filter numerator and denominator.
        self.filt_num =  (1 - self.parameters.alpha) * self.filt_num + \
                self.parameters.alpha * (self.gauss_fd * np.conjugate(np.fft.fft2(img_mov)))
        self.filt_denom = (1 - self.parameters.alpha) * self.filt_denom + \
                self.parameters.alpha * (np.fft.fft2(img_mov) * np.conjugate(np.fft.fft2(img_mov)))
        
        if self.parameters.measure_runtime:
            self.total_track_time += time.time() - time_start
        
        # Return new estimated position.
        return self.pos


    def _construct_filter(self, img_ext, gauss_fd):
        """
        Construct correlation filter that best matches provided ideal response.

        Args:
            img_ext (numpy.ndarray): Region of interest extracted form image.
            gauss_fd (numpy.ndarray): Frequency domain representation of the ideal filter response.

        Returns:
            (tuple): Filter numberator and denominator.
        """

        # Preprocess image
        img_prep = self._preprocess(img_ext)

        # Compute filter numerator and denominator.
        filt_num = gauss_fd * np.conjugate(np.fft.fft2(img_prep))
        img_ext_fd = np.fft.fft2(img_ext)
        filt_denom = img_ext_fd * np.conjugate(img_ext_fd)

        # Perform additional training by warping the image and
        # updating the numerator and denominator of the filter.
        for _ in range(self.parameters.training_iter):
            if self.parameters.rotate:
                img_prep = self._preprocess(self._warp(img_ext))
            else:
                img_prep = self._preprocess(img_ext)
            filt_num = filt_num + gauss_fd * np.conjugate(np.fft.fft2(img_prep))
            filt_denom = filt_denom + np.fft.fft2(img_prep) * np.conjugate(np.fft.fft2(img_prep))

        # Return filter numerator and denominator.
        return filt_num, filt_denom


    def _gauss_image(self, img, gt, sig):
        """
        Compute ideal correlation filter response on image containing the object of interest.
        
        Args:
            img (numpy.ndarray): Image containing the object of interest.
            gt (list): Bounding box specifiying the region of interest.
            sig (float): Sigma parameter for the gaussian distribution.

        Returns:
            (numpy.ndarray): Image containing ideal gaussian response.
        """

        # Compute ideal filter response at region of interest in image.
        xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        resp = np.exp(-(((xx - (gt[0] + gt[2] * 0.5))**2 + (yy - (gt[1] + gt[3] * 0.5))**2) / (2 * sig)))
        return resp

    
    def _preprocess(self, img):
        """
        Normalize image and apply 2d Hanning window.

        Args:
            img (numpy.ndarray): Image to preprocess.

        Returns:
            (numpy.ndarray): Normalized image with applied window.

        """

        # Compute 2d Hanning window.
        win = np.multiply(*np.meshgrid(np.hanning(img.shape[1]), np.hanning(img.shape[0])))

        # Normalize image.
        img_log = np.log(img + 1)
        img_n = (img_log - np.mean(img_log)) / (np.std(img_log) + 1e-5)
        
        # Apply window to image.
        return win*img_n
   

    def _warp(self, img):
        """
        Warp image using rotation matrix.

        Args:
            img (numpy.ndarray): Image to randomly rotate.
        """
        
        # Compute rotational angle.
        rot = -180/16 + (180/16 + 180/16) * np.random.uniform()

        # Get rotational matrix and apply rotation.
        matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rot, 1)
        img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0])).astype(np.float32)

        # Return rotated image.
        return img_rot


