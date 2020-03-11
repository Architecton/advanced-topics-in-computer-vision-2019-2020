import numpy as np
import matplotlib.pyplot as plt
from ex1_utils import rotate_image, show_flow
from lucas_kanade import lucas_kanade
from horn_schunck import horn_schunck
from iterative_lucas_kanade import iterative_lucas_kanade


# Construct test images.
im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = rotate_image(im2, -1)

# from PIL import Image
# im1 = np.array(Image.open("../disparity/cporta_left.png")).astype(float)/255.0
# im2 = np.array(Image.open("../disparity/cporta_right.png")).astype(float)/255.0

# # Lucas-Kanade method results.
U_lk, V_lk = lucas_kanade(im1, im2, 3, derivative_smoothing=True)

# Iterative Lucas-Kanade method results.
U_lk_it, V_lk_it = iterative_lucas_kanade(im1, im2, 3, derivative_smoothing=True)

# # Horn-Schunck method results.
U_hs, V_hs = horn_schunck(im1, im2, n_iters=1000, lmbd=0.5)
# 
# # Horn-Schunck with Lucas-Kanade initialization results.
U_hs_init, V_hs_init = horn_schunck(im1, im2, n_iters=1000, lmbd=0.5, u_init=U_lk, v_init=V_lk)


### Construct and show plots. ###

# # Lucas-Kanade
fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_lk, V_lk, ax1_21, type='angle')
show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
fig1.suptitle('Lucas−Kanade Optical Flow')
 
# Horn-Schunck
fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
ax2_11.imshow(im1)
ax2_12.imshow(im2)
show_flow(U_hs, V_hs, ax2_21, type='angle')
show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
fig2.suptitle('Horn−Schunck Optical Flow')

# Iterative Lucas-Kanade
fig3, ((ax3_11, ax3_12), (ax3_21, ax3_22)) = plt.subplots(2, 2)
ax3_11.imshow(im1)
ax3_12.imshow(im2)
show_flow(U_lk_it, V_lk_it, ax3_21, type='angle')
show_flow(U_lk_it, V_lk_it, ax3_22, type='field', set_aspect=True)
fig3.suptitle('Iterative Lucas−Kanade Optical Flow')

# Horn-Schunck with Lucas-Kanade initialization.
fig4, ((ax4_11, ax4_12), (ax4_21, ax4_22)) = plt.subplots(2, 2)
ax4_11.imshow(im1)
ax4_12.imshow(im2)
show_flow(U_hs_init, V_hs_init, ax4_21, type='angle')
show_flow(U_hs_init, V_hs_init, ax4_22, type='field', set_aspect=True)
fig4.suptitle('Horn−Schunck with Lucas−Kanade Initialization Optical Flow')

# Show plot.
plt.show()

