import numpy as np
import random
import cv2
from tracker import Tracker
from matrices import get_fi_Q
from particle_filter_utils import evaluate_likelihood, get_ref


class BPFParams():

    def __init__(self, num_particles=200, dyn_model='ncv', comp_method='hist-hsv'):
        self.num_particles = num_particles
        self.comp_method = comp_method

        if dyn_model == 'ncv':
            # NEAR CONSTANT VELOCITY MODEL
            f = np.array([[0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]])
            l = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
            fi, Q = get_fi_Q(f, l)
            self.fi = fi
            self.Q = Q

            # Set measurement length.
            self.meas_len = 6
        else:
            # TODO
            pass


class BPFTracker(Tracker):


    def __init__(self, parameters=BPFParams()):
        self.parameters = parameters
 

    def name(self):
        return "BPF Tracker"


    def initialize(self, image, region, sequence):

        # get initial image and ground truth (bounding box)
        img_init = image # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.init_gt = np.array(region).astype(int)
        
        # Set initial position and initial inbound position.
        self.pos = self.init_gt.copy()
        self.inb_pos = np.array([self.pos[0], self.pos[1], self.pos[0] + self.pos[2], self.pos[1] + self.pos[3]]).astype(np.int64)
        
        # Get reference model and comparison function.
        self.ref, self.obs_func = get_ref(image, region, 'hist-hsv', compare_method=cv2.HISTCMP_BHATTACHARYYA)
        
        # Set state transition matrix and covariance matrix Q.
        q = 1.0 # 3/4*1.0**2
        self.Q = np.array(self.parameters.Q.subs({'q':q})).astype(float)

        self.fi = np.array(self.parameters.fi).astype(float)

        # Initialize particles and their weights.
        self.particles = [np.repeat(np.zeros((1, self.parameters.meas_len), dtype=float), self.parameters.num_particles, axis=0),
                          1/self.parameters.num_particles*np.ones(self.parameters.num_particles, dtype=float)]

        # Sample and set initial positions.
        INIT_SCALE = 2.0
        self.particles[0][:, 0:4] = np.random.multivariate_normal((self.inb_pos[0], self.inb_pos[1], self.pos[2], self.pos[3]),
                                                                  INIT_SCALE*np.eye(4), self.particles[0].shape[0])

        self.sequence = sequence


    def track(self, image):
        
        
        ### II. SAMPLING NEW PARTICLES ###
        particles_sampled = np.vstack(random.choices(population=self.particles[0],
                                      weights=self.particles[1], k=self.particles[0].shape[0]))
        
        ### III. APPLYING THE DYNAMIC MODEL TO THE PARTICLES ###
        # FOR NOW
        particles_fi = np.matmul(self.fi, particles_sampled.transpose()).transpose() + \
                 np.random.multivariate_normal(np.zeros(self.Q.shape[0], dtype=float), self.Q, particles_sampled.shape[0])

        # particles_fi = particles_sampled
        # particles_fi += np.random.multivariate_normal(np.zeros(self.Q.shape[0], dtype=float), self.Q, particles_sampled.shape[0])

        particles_new_fi = [particles_fi, np.empty(self.particles[0].shape[0], dtype=float)]
        
        particles_new_obs = evaluate_likelihood(image, particles_new_fi, self.obs_func, self.ref)
        self.particles = particles_new_obs
        


        # TODO update reference model.
        best_particle = particles_new_obs[0][np.argmax(particles_new_obs[1]), :]

        mean_particle = np.sum(particles_new_obs[0]*particles_new_obs[1][:, np.newaxis], axis=0)
        return mean_particle[:4], particles_new_obs
        





