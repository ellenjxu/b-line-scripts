"""
Active snake contour for image segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2 as cv

image = cv.imread("C:/Users/ellen/Documents/code/B-line_detection/scripts/clip_20.png")

# circle
# s = np.linspace(0, 2*np.pi, 400)
# r = 200 + 300*np.sin(s)
# c = 275 + 300*np.cos(s)

# ultrasound

# top curve
s = np.linspace(1/8*np.pi, 7/8*np.pi, 400) # theta
r = -50 + 75*np.sin(s) # y
c = 270 + 100*np.cos(s) # x

# left line
r2 = 340
c2 = 0

# bottom curve
s3 = np.linspace(7/8*np.pi, 1/8*np.pi, 400)
r3 = 250 + 175*np.sin(s3)
c3 = 270 + 300*np.cos(s3)

# right line
r4 = 0
c4 = 340

init = np.array([r, c]).T
init2 = np.array([r2, c2]).T
init3 = np.array([r3, c3]).T
init4 = np.array([r4, c4]).T
init = np.vstack([init, init2, init3, init4])

print(init)
snake = active_contour(gaussian(image, 3, preserve_range=False), init, alpha=0.015, beta=10, gamma=0.001)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, image.shape[1], image.shape[0], 0])
plt.show()