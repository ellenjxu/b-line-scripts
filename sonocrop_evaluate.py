"""
Compare automated sonocrop processing with .pkl corner point annotation. Use IOU/Euclidean distance to evaluate.
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from shapely.geometry import Polygon
from sonocrop_corner_points import get_corners_simple, get_corners_canny
from sonocrop_mask import get_mask
from sonocrop.vid import get_edges

# get corner points

def get_corner_points(input_file, display=False):

    """
    Get corner points from .pkl (red) and sonocrop (blue).
    """

    corner_points_path = "C:/Users/ellen/Documents/code/B-line_detection/supplementary_files/corner_points_dictionary.pkl"
    points_dict = pd.read_pickle(corner_points_path)

    # get corner points from sonocrop
    mask = get_mask(input_file)
    top, bottom, left, right = get_edges(input_file, thresh=0.05)   # 0.05
    mask = mask[top:bottom, left:right]

    corner_points_auto = get_corners_simple(mask)
    corner_points_auto = [list(x) for x in corner_points_auto]  # turn tuple into list
    corner_points_auto[2], corner_points_auto[3] = corner_points_auto[3], corner_points_auto[2] # swap the points
    # print(corner_points_auto)   # tl, bl, br, tr

    # get corner points from .pkl

    file_name = input_file.split("/")[-1].split(".")[0]

    _, case, clip = os.path.splitext(file_name)[0].split('_')
    case_name = 'Case-'+case
    corner_points = points_dict[case_name][clip]
    corner_points = [list(x) for x in corner_points]    # turn tuple into list
    corner_points = [[int(x[0])-left, int(x[1])-top] for x in corner_points]    # adjust for shift

    for point in corner_points:
        if point[0] < 0:
            point[0] = 0
        if point[1] < 0:
            point[1] = 0

    corner_points[2], corner_points[3] = corner_points[3], corner_points[2] # swap the points
    corner_points = [tuple(x) for x in corner_points]   # turn list into tuple
    # print(corner_points)    # tl, bl, br, tr

    # compare corner points on mask

    if display:
        plt.imshow(mask, cmap='gray')
        plt.scatter(*zip(*corner_points), c='r')
        plt.scatter(*zip(*corner_points_auto), c='b')
        plt.show()

    return corner_points, corner_points_auto

def evaluate(input_file, corner_points, corner_points_auto, is_iou=True):

    """
    IOU and Euclidean distance evaluation of corner points (.pkl = red, sonocrop = blue).
    """

    vid = cv.VideoCapture(input_file)
    _,img = vid.read()

    # cv.rectangle(img, corner_points[0], corner_points[3], (255, 0, 0), 2)
    # cv.rectangle(img, corner_points_auto[0], corner_points_auto[3], (0, 0, 255), 2)

    pts = np.array(corner_points, np.int32)
    pts = pts.reshape((-1,1,2))

    pts_auto = np.array(corner_points_auto, np.int32)
    pts_auto = pts_auto.reshape((-1,1,2))

    # cv.polylines(img, [pts], True, (255, 0, 0), 2)
    # cv.polylines(img, [pts_auto], True, (0, 0, 255), 2)
    # cv.imshow('Bounding trapezoids', img)
    # cv.waitKey(0)

    def iou(pts1, pts2):
        poly1 = Polygon(pts1)
        poly2 = Polygon(pts2)

        intersection = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - intersection

        return intersection/union

    if is_iou:
        iou = iou(corner_points, corner_points_auto)
        return iou

    # euclidean distance

    def euclidean_distance(pts1, pts2):
        distance = 0
        for i in range(len(pts1)):
            distance += np.sqrt((pts1[i][0] - pts2[i][0])**2 + (pts1[i][1] - pts2[i][1])**2)
        return distance

    if not is_iou:
        euclidean_distance = euclidean_distance(corner_points, corner_points_auto)
        return euclidean_distance

# loop over all files

def main():
  root = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/"

  for i in range(1,123):

    folder = root + "Case-" + str(i).zfill(3)

    if os.path.exists(folder):
        for file in os.listdir(folder):

            if file.endswith(".mp4") and not file.endswith("masked.mp4"):
                input_file = os.path.join(folder, file)

                try:
                    corner_points, corner_points_auto = get_corner_points(input_file, display=False)
                    iou = evaluate(input_file, corner_points, corner_points_auto, is_iou=True)
                    
                    # write to text file
                    with open("C:/Users/ellen/Documents/code/B-line_detection/scripts/results/iou_results.txt", "a") as f:
                        f.write(input_file + " " + str(iou) + "/n")
                    
                    print(input_file + " " + str(iou))
                    break

                except:
                    print("Error:", input_file)
                    continue

            else:
                continue

# main()

# test
# file = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-111/BEDLUS_111_001.mp4"
file = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/LUS_videos/Case-122/BEDLUS_122_011.mp4"
corner, corner_auto = get_corner_points(file, display=True)
# evaluate(file, corner, corner_auto, is_iou=True)