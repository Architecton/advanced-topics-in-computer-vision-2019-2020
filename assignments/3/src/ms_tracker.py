import numpy as np
import cv2

from ex2_utils import Tracker, create_epanechnik_kernel, get_patch, extract_histogram, backproject_histogram

class MeanShiftTracker(Tracker):
    """
    Implementation of the mean-shift tracking algorithm.
    Author: Jernej Vivod

    Args:
        params (obj): MSParams instance specifying the parameters for the tracker.
    """
    
    def initialize(self, img, region):
        """
        Initialize the mean-shift tracker.

        Args:
            img (numpy.ndarray): First image.
            region (list): bounding box specification for the
            object on the first image. First and second values
            represent the position of the left-upper corner. The
            third and fourth values represent the width and the 
            height of the bounding box respectively.
        """

        
        # Set tracker name.
        self.name = "mean-shift-tracker"
        
        # Number of iterations performed to reposition bounding box,
        # maximum number of iterations performed and umber of trackings performed.
        self.num_it = []
        self.max_it = 0
        self.num_tracking_runs = 0

        # Get initialized position.
        self.pos = [region[0] + region[2] / 2, region[1] + region[3] / 2]

        # Set tracking patch size. Increment size by one if even.
        self.size = (int(region[2]) + abs(int(region[2]) % 2 - 1), int(region[3]) + abs(int(region[3]) % 2 - 1))
        
        # Initialize tracking window indices grid.
        self.mesh_x, self.mesh_y = np.meshgrid(np.arange(-self.size[0]//2+1, 
            self.size[0]//2+1), np.arange(-self.size[1]//2+1, self.size[1]//2+1))
        
        # Initialize kernels.
        self.kern1 = create_epanechnik_kernel(self.size[0], self.size[1], 2)
        self.kern2 = np.ones((self.size[1], self.size[0]))
        self.kern_bandwidth = 4

        # Get initial patch.
        patch = get_patch(img, self.pos, self.size)
        
        # Extract hitrogram from template using the specified kernel.
        hist_ = extract_histogram(patch[0], self.parameters.n_bins, weights=self.kern1)
        self.hist = hist_/np.sum(hist_)
        

    def track(self, img):
        """
        Perform tracking on next image using reference color histogram model.

        Args:
            img (numpy.ndarray): Image on which to localize the object
            using the reference model.

        Returns:
            (list): bounding box specification for the
            object on the first image. First and second values
            represent the position of the left-upper corner. The
            third and fourth values represent the width and the 
            height of the bounding box respectively.
        """
        
        # Initialize convergence flag.
        convergence_flg = False
        
        # Initialize iteration counter.
        num_it = 0

        # Repeat until convergence or until maximum number of iterations
        # exceeded.
        while not convergence_flg and num_it < self.parameters.max_it:
            
            # Increment iteration counter.
            num_it += 1

            # Extract histogram and current location.
            patch = get_patch(img, self.pos, self.size)
            hist_ = extract_histogram(patch[0], self.parameters.n_bins, weights=self.kern1)
            hist_nxt = hist_/np.sum(hist_)

            # Compute the weights w_{i}.
            weights = np.sqrt(self.hist/(hist_nxt + 1.0e-4))

            # Backproject within extracted patch using weights v.
            bp = backproject_histogram(patch[0], weights, self.parameters.n_bins)

            # Get changes in x and y directions.
            delta_x = np.sum(self.mesh_x*bp)/np.sum(bp)
            delta_y = np.sum(self.mesh_y*bp)/np.sum(bp)

            # Check if division successful.
            if np.isnan(delta_x) or np.isnan(delta_y):
                break

            # If changes sufficiently small or if maximum number of iterations exceeded.
            if abs(delta_x) < 1.0 and abs(delta_y) < 1.0:
                # Set convergence flag.
                convergence_flg = True

                # Increment number of total iterations and trackings.
                self.num_it.append(num_it)
                self.num_tracking_runs += 1
                
                # If new maximum of iteration number observed.
                if num_it > self.max_it:
                    self.max_it = num_it
            else:
                # Add changes in x and y direction to current position.
                self.pos[0] += np.round(delta_x)
                self.pos[1] += np.round(delta_y)
        
        # Update reference model with current model.       
        self.hist = (1-self.parameters.alpha)*self.hist + self.parameters.alpha*hist_nxt 

        # Return found position.
        return [self.pos[0] - self.size[0]/2, self.pos[1] - self.size[1]/2, self.size[0], self.size[1]]



class MSParams():
    """
    Encapsulation of the mean-shift tracking algorithm parameters. 
    
    Args:
        max_it (int): Maximum iterations to perform when performing
        mode seeking using the mean shift algorith.
        n_bins (int): Length of side of 3D color histogram.
        alpha (float): Weight given to current model in influencing the reference model.
    """

    def __init__(self, max_it, n_bins, alpha):
        self.max_it = max_it
        self.n_bins = n_bins
        self.alpha = alpha

