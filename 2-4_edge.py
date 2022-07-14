"""
Experimenting with OpenCV edge detection
"""

import cv2

# Read the original image
img = cv2.imread("C:/Users/ellen/Documents/code/B-line_detection/scripts/clip_20.png") 

# Tutorial

# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=50) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Custom filter

# import numpy as np

# kernel = np.array([[0, 1, 2],
#                    [-1, 0, 1],
#                    [-2, -1, 0]])

# dst = cv2.filter2D(img, -1, kernel)
# cv2.imshow('Custom', dst)
# cv2.waitKey(0)

# dst = cv2.filter2D(img, -1, kernel.T)
# cv2.imshow('Custom', dst)
# cv2.waitKey(0)