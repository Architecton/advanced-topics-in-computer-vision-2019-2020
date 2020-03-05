import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio

from ex1_utils import gausssmooth
from lucas_kanade import lucas_kanade


def superimpose_field(u, v, img):
    """
    Superimpose quiver plot onto specified image and return result as numpy array
    
    Args:
        u (numpy.ndarray): Delta values in the x direction.
        v (numpy.ndarray): Delta values in the y direction.
        img (numpy.ndarray): Image on which to superimpose the quiver plot.
    
    Returns:
        (numpy.ndarray): Resulting image in the form of a numpy array.
    """
    
    # Set scaling.
    scaling = 0.1
    u = cv2.resize(gausssmooth(u, 1.5), (0, 0), fx=scaling, fy=scaling)
    v = cv2.resize(gausssmooth(v, 1.5), (0, 0), fx=scaling, fy=scaling)
    
    # Normalize magnitudes.
    u = u / np.sqrt(u**2 + v**2);
    v = v / np.sqrt(u**2 + v**2);
    
    # Create plot.
    x_ = (np.array(list(range(1, u.shape[1] + 1))) - 0.5) / scaling
    y_ = (np.array(list(range(1, u.shape[0] + 1))) - 0.5) / scaling
    x, y = np.meshgrid(x_, y_)
    fig = plt.figure()
    ax = plt.gca()
    ax.axis('off')
    ax.quiver(x, y, u, v, color='r')
    ax.imshow(img)
    fig.canvas.draw()
    plt.close()
    
    # Get plot in shape of numpy array and return it.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        

def visualize_flow(src, flow_comp_func):
    """
    Visualize optical flow on a sequence of image frames. The
    optical flow is computed using the specified function.

    Author: Jernej Vivod (vivod.jernej@gmail.com)

    Args:
        src (str): Path to video file for which to compute and
        visualize the optical flow.

        flow_comp_func (function): Function implementing
        an optical flow computation method.

    Returns:
        None
    """

    # Flag specifying whether this is the first frame or not.
    first_frame = True
    
    # Initialize video stream.
    video_capture = cv2.VideoCapture(src)
    if not video_capture.isOpened():
        raise RuntimeError("Error opening video stream.")
    
    # Previous frame buffer and results array.
    prev = None
    res = []
    
    # Set frame counter.
    count = 0
    FRAME_LIM = 100
    
    # While there is video and while below frame index limit.
    while video_capture.isOpened() and count < FRAME_LIM:

        # Increment frame count.
        count += 1

        # Get next frame.
        ret, frame = video_capture.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # If first frame ...
            if first_frame:
                
                # Set frame as previous frame.
                prev = frame_gray
                prev_original = frame
                
                # Set first frame flag to false.
                first_frame = False

            else:

                # Compute flow using current frame and frame
                # in previous buffer.
                u, v = flow_comp_func(prev.astype(float)/255.0, frame_gray.astype(float)/255.0)

                # Add optical flow visualization to image in prev buffer.
                vis_nxt = superimpose_field(u, v, prev_original)

                prev = frame_gray
                prev_original = frame

                # Store visualization in results buffer
                res.append(vis_nxt)

        else:
            break
    
    # Release stream.
    video_capture.release()

    # Create gif file from images.
    imageio.mimsave('test.gif', res)


### TEST ###
if __name__ == '__main__':
    visualize_flow('../data/chaplin.mp4', lambda im1, im2: lucas_kanade(im1, im2, n=3))

