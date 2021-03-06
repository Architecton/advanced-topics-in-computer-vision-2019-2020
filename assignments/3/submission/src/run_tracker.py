import time
import cv2
import imageio
import argparse

from sequence_utils import VOTSequence
from mosse_tracker import MosseTracker, MosseParams

# Parse name of sequence for which to perform tracking as well as tracking parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--seq', type=str, default='car')
parser.add_argument('--save-gif', action='store_true', default=False)

### Parse tracker parameters ###
parser.add_argument('--lmbd', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.125)
parser.add_argument('--sigma', type=float, default=20)
parser.add_argument('--training-iter', type=int, default=0)
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--measure-runtime', action='store_true')

args = parser.parse_args()

# Set the path to directory where sequences are stored.
dataset_path = '../data/vot2014'

# Select sequence on which to test.
sequence_name = args.seq

# Set visualization and setup parameters.
vis_params = {
    'win_name' : 'Tracking window',
    'reinitialize' : True,
    'show_gt' : False,
    'video_delay' : 15,
    'font' : cv2.FONT_HERSHEY_PLAIN
}

# Create a sequence object.
sequence = VOTSequence(dataset_path, sequence_name)

# Set initialization frame to first frame.
init_frame = 0

# Initialize failure counter to 0.
n_failures = 0

# Initialize tracker.
parameters = MosseParams(lmbd=args.lmbd, alpha=args.alpha, sig=args.sigma, training_iter=args.training_iter, rotate=args.rotate, scale=args.scale, measure_runtime=args.measure_runtime)
tracker = MosseTracker(parameters)

# Initialize running time.
time_all = 0

# If visualizing results.
if args.visualize:
    # Initialize visualization window.
    sequence.initialize_window(vis_params["win_name"])

# tracking loop - goes over all frames in the video sequence
frame_idx = 0

# While sequence not finished.
while frame_idx < sequence.length():

    # Get next frame from sequence.
    img = cv2.imread(sequence.frame(frame_idx))

    # If initialization frame, initialize tracker.
    if frame_idx == init_frame:
        
        # Initialize tracker (at the beginning of the sequence or after tracking failure).
        t_ = time.time()
        tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
        time_all += time.time() - t_
        predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
    else:
        # Track on current frame - predict bounding box.
        t_ = time.time()
        predicted_bbox = tracker.track(img)
        time_all += time.time() - t_

    # calculate overlap with prediction and ground truth (needed to determine failure of a tracker).
    gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
    o = sequence.overlap(predicted_bbox, gt_bb)

    # Draw ground-truth and predicted bounding boxes, frame numbers and show image.
    # If drawing ground truth boundign box.
    if vis_params["show_gt"]:
        sequence.draw_region(img, gt_bb, (0, 255, 0), 1)

    # Draw region and data.
    sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
    sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
    sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
    
    # If visualizing, show image.
    if args.visualize:
        sequence.show_image(img, vis_params["video_delay"])
    
    # Save image if creating gif file.
    if args.save_gif:
        sequence.save_image(img)
    
    # If some overlap or no reinitialization after failure.
    if o > 0 or not vis_params["reinitialize"]:
        # Increase frame counter by 1 (tracking successful).
        frame_idx += 1
    else:
        # Increase frame counter by 5 and set re-initialization to the next frame.
        frame_idx += 5
        init_frame = frame_idx

        # Increment failure counter.
        n_failures += 1


# Print tracker statistics to file.
with open("../results/results.txt", "a") as f:
    f.write("{0} ({1}): {2}\n".format(sequence_name, tracker.name(), n_failures))

# If creating gif file, create it and save.
if args.save_gif:
    imageio.mimsave('../results/' + sequence_name + '.gif', sequence.result, duration='0.03')

if args.measure_runtime:
    print("mean initialization time: {0}".format(tracker.total_init_time/tracker.init_count))
    print("mean track step time: {0}".format(tracker.total_track_time/tracker.track_count))

# Print tracker statistics.
print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))

