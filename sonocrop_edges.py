"""
Get edges from .mp4.
"""

from sonocrop import vid
from sonocrop.vid import get_edges
from sonocrop_mask import get_mask
import cv2 as cv
from matplotlib import pyplot as plt

# test cases

# C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-122/BEDLUS_122_011.mp4
# C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-103/BEDLUS_103_001.mp4
# C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-052/test.mp4
input_file = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-001/BEDLUS_001_001.mp4"
top, bottom, left, right = get_edges(input_file, thresh=0.05)
print(top, bottom, left, right)

# vid = cv.VideoCapture(input_file)
# _,img = vid.read()

# cv.imshow('Image', img)
# cv.waitKey(0)

# crop mask on edges
mask = get_mask(input_file)
mask = mask[top:bottom, left:right]
plt.imshow(mask, cmap='gray')
plt.show()