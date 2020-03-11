import cv2
import numpy as np
from imageio import imread

import pdb
pdb.set_trace()

im1 = np.asarray(imread("0.png"))
im2 = np.asarray(imread("1.png"))
# im1 = np.float64(im1 / 255)
# im2 = np.float64(im2 / 255)

flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    
    import pdb
    pdb.set_trace()

    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

hsv = draw_hsv(flow)
im2w = warp_flow(im1, flow)
cv2.imwrite("/tmp/flow.jpg",hsv)
cv2.imwrite("/tmp/im1.jpg", im1*255)
cv2.imwrite("/tmp/im2.jpg", im2*255)
cv2.imwrite("/tmp/im2w.jpg", im2w)

