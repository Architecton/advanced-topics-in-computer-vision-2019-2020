import numpy as np
from timeit import default_timer as timer

from ex1_utils import rotate_image
from lucas_kanade import lucas_kanade
from horn_schunck import horn_schunck


def measure_runtime(im1, im2, flow_comp_func, num_repetitions):
    """
    Measure average runtime of specified optical flow computation algorithm
    implementation. Measure and return the average runtime for specified
    number of repetitions.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        im1 (np.ndarray): First frame.
        im2 (np.ndarray): First frame.
        flow_comp_func (function): Function implementing an optical flow computation method.
        num_repetitions (int): Number of repetitions of the flow computation algorithm
        implementation to perform.

    Returns:
       (float): Average runtime for the specified optical flow computation method implementation
       over the specified number of repetitions.
    """

    # Initialize runtime accumulator.
    time_acc = 0.0

    # Perform num_repetitions repetitions of the specified
    # optical flow computation algorithm.
    for idx in np.arange(num_repetitions):

        # Time execution of specified optical flow
        # computation algorithm.
        time_start = timer()
        flow_comp_func(im1, im2)
        time_acc = timer() - time_start
    
    # Return average runtime.
    return time_acc/num_repetitions


if __name__ == '__main__':
    # Perform tests.

    # Initialize frames.
    im1 = np.random.rand(200, 200 ).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)
    
    # Set number of repetitions for the evaluation.
    NUM_REPETITIONS = 100
    
    # Initialize optical flow estimation methods and measure runtime.
    flow_comp_func1 = lambda im1, im2: lucas_kanade(im1, im2, n=3, derivative_smoothing=True)
    flow_comp_func2 = lambda im1, im2: horn_schunck(im1, im2, n_iters=1000, conv=False, lmbd=0.5, derivative_smoothing=True)
    flow_comp_func3 = lambda im1, im2: horn_schunck(im1, im2, n_iters=1000, conv=True, lmbd=0.5, derivative_smoothing=True)
    res1 = measure_runtime(im1, im2, flow_comp_func1, NUM_REPETITIONS)
    res2 = measure_runtime(im1, im2, flow_comp_func2, NUM_REPETITIONS)
    res3 = measure_runtime(im1, im2, flow_comp_func3, NUM_REPETITIONS)

    print("Average runtime for Lucas-Kanade ({0} repetitions): {1}s".format(NUM_REPETITIONS, res1))
    print("Average runtime for Horn-Schunck (no convergence criterion) ({0} repetitions): {1}s".format(NUM_REPETITIONS, res2))
    print("Average runtime for Horn-Schunck (convergence criterion) ({0} repetitions): {1}s".format(NUM_REPETITIONS, res3))
    

