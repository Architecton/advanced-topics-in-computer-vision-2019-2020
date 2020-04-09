import matplotlib.pyplot as plt
import cv2
import argparse

import numpy as np
from ex2_utils import generate_responses_1, generate_responses_2

# Parse dataset argument.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=1, choices=[1, 2])
args = parser.parse_args()

# Set parameters.
WINDOW_SIZE = (9, 9) if args.dataset == 1 else (101, 101)
KERNEL = np.ones(WINDOW_SIZE)
KERN_BANDWIDTH = 4
RESP = generate_responses_1() if args.dataset == 1 else generate_responses_2()

def onclick(event, resp, window_size, kernel, kern_bandwidth):

    # If clicked position on plot.
    if event.xdata != None and event.ydata != None:

        # Get clicked position.
        pos = [abs(int(round(event.xdata))), abs(int(round(event.ydata)))]

        # Create mesh grid representing indices in window.
        mesh_x, mesh_y = np.meshgrid(np.arange(-window_size[0]//2+1, window_size[0]//2+1), 
            np.arange(-window_size[1]//2+1, window_size[1]//2+1))

        # Get PDF values in window.
        vals = resp[pos[1]-window_size[1]//2:min(pos[1]+window_size[1]//2+1, resp.shape[1]), 
                pos[0]-window_size[0]//2:min(pos[0]+window_size[0]//2+1, resp.shape[0])]

        # Visualize starting position (draw rectangle).
        vis = cv2.cvtColor(255*(resp/np.max(resp)), cv2.COLOR_GRAY2RGB).astype(np.uint8)
        vis = cv2.rectangle(vis, (pos[0]-window_size[1]//2, pos[1]-window_size[1]//2), 
                (pos[0]+window_size[0]//2+1, pos[1]+window_size[0]//2+1), (255, 0, 0), 1 if args.dataset == 1 else 8)
        plt.clf()
        plt.imshow(vis, cmap='gray')
        plt.draw()
        
        # Initialize convergence flag, set maximum number of iterations,
        # initialize iteration counter and set kernel bandwidth.
        convergence_flg = False
        max_it = 20 if args.dataset == 1 else 2000
        num_it = 0
        
        # While convergence not declared.
        while not convergence_flg:

            # Increment iteration counter.
            num_it += 1
            
            # Get changes in x and y directions.
            delta_x = np.sum(mesh_x*vals)/np.sum(vals)
            delta_y = np.sum(mesh_y*vals)/np.sum(vals)

            delta_x *= 3
            delta_y *= 3

            # Check if division successful.
            if np.isnan(delta_x) or np.isnan(delta_y):
                break

            # If changes sufficiently small or if maximum number of iterations exceeded.
            if abs(delta_x) < (1.0e-1 if args.dataset == 1 else 1.0) and abs(delta_y) < (1.0e-1 if args.dataset == 1 else 1.0) or num_it >= max_it:

                print("Converged in {0} iterations.".format(num_it))

                # Set convergence flag.
                convergence_flg = True

                # Draw rectangle at final position.
                vis = cv2.rectangle(vis, (pos[0]-window_size[1]//2, pos[1]-window_size[1]//2), 
                        (pos[0]+window_size[0]//2+1, pos[1]+window_size[0]//2+1), (0, 0, 255), 1 if args.dataset == 1 else 8)
                plt.clf()
                plt.imshow(vis)
                plt.draw()

            else:

                # Add changes in x and y direction to current position.
                pos[0] += int(np.sign(delta_x)*np.ceil(abs(delta_x)))
                pos[1] += int(np.sign(delta_y)*np.ceil(abs(delta_y)))

                # Get PDF values in moved window.
                vals = resp[pos[1]-window_size[1]//2:min(pos[1]+window_size[1]//2+1, resp.shape[1]), 
                        pos[0]-window_size[0]//2:min(pos[0]+window_size[0]//2+1, resp.shape[0])]


# Plot response and add click callback.
fig, ax = plt.subplots()
plt.imshow(RESP, cmap='gray')
cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, resp=RESP, window_size=WINDOW_SIZE, kernel=KERN_BANDWIDTH, kern_bandwidth=KERN_BANDWIDTH))
plt.show()
plt.draw()

