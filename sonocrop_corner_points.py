"""
Get cornerpoints from sonocrop mask.
"""

# from sonocrop.cli import get_mask
from sonocrop_mask import get_mask
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# ---------------------------------- simple test -----------------------------------

# file = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/images/example/BEDLUS_103_001_mask.png"

def get_corners_simple(mask):

    """
    Split image in half and get the highest pixel and leftmost pixel on each half as corners.
    """

    # img = cv.imread(file)
    # img = Image.fromarray(mask)
    arr = np.array(mask, dtype=np.uint8)
    # arr = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    lhalf = arr[:,0:arr.shape[1]//2]
    rhalf = arr[:,arr.shape[1]//2:]

    def get_topleft(arr):
        for i in range(lhalf.shape[0]): # row
            for j in range(lhalf.shape[1]): # column
                if lhalf[i][j] != 0:
                    point = (j,i)
                    return point

    def get_bottomleft(arr):
        for i in range(lhalf.shape[1]):  # column
            for j in range(lhalf.shape[0]):  # row
                if lhalf[j][i] != 0:
                    point = (i,j)
                    return point

    def get_topright(arr):
        for i in range(rhalf.shape[0]):  # row
            for j in range(rhalf.shape[1]):  # column
                if rhalf[i][j] != 0:
                    point = (arr.shape[1]//2+j,i)
                    return point

    def get_bottomright(arr):
        for i in range(rhalf.shape[1]-1, 0, -1):  # column
            for j in range(rhalf.shape[0]):  # row
                if rhalf[j][i] != 0:
                    point = (arr.shape[1]//2+i,j)
                    return point

    return get_topleft(arr), get_bottomleft(arr), get_topright(arr), get_bottomright(arr)

# ------------------------------ canny + hough method ------------------------------

def get_corners_canny(file):

    """
    Use canny edge detection and hough probabilistic to find the corners.
    """

    img = cv.imread(file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    edges = cv.Canny(gray, 1000, 1200, apertureSize=3) # 50 150
    cv.imshow('Canny Edge Detection', edges)
    cv.waitKey(0)
    
    lines_list = []
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=150, maxLineGap=200)
    
    for points in lines:
        x1,y1,x2,y2=points[0]
        img1 = cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        lines_list.append([(x1,y1),(x2,y2)])

    # Show the result img
    cv.imshow('Detected lines',img1)
    cv.waitKey(0)

    # Display all points
    for lines in lines_list:
        point1 = lines[0]
        point2 = lines[1]
        plt.plot(point1[0],point1[1],'ro')
        plt.plot(point2[0],point2[1],'ro')

    plt.show()

    # Get two highest and two lowest y values for corners
    y_values = []
    for lines in lines_list:
        point1 = lines[0]
        point2 = lines[1]
        y_values.append(point1[1])
        y_values.append(point2[1])

    sort_index = np.argsort(y_values, axis=0)
    get_index = sort_index[:2]
    get_index = np.append(get_index, sort_index[-2:])

    # Get corner points
    corner_points = []
    for i in range(len(get_index)):
        index1 = get_index[i]//2
        index2 = get_index[i]%2
        # print(index1, index2)
        corner_points.append(lines_list[index1][index2])

    print(corner_points)

    # Draw on original image
    for points in corner_points:
        img2 = cv.circle(img,points,radius=3, color=(0, 0, 255), thickness=-1)

    # Show the result img
    cv.imshow('Corner points',img2)
    cv.waitKey(0)

# ------------------------------ run ------------------------------

# input_file = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-122/BEDLUS_122_011.mp4"
# # get_corners_simple(get_mask(input_file))

# # plot on the image
# # plt.imshow(arr, cmap='gray')
# # plt.show()

# img = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-122/BEDLUS_122_011_mask.png"

# for point in get_corners_simple(input_file):
#     cv.circle(img,point,radius=3, color=(0, 0, 255), thickness=-1)

# # Show the result img
# cv.imshow('Corner points', img)
# cv.waitKey(0)