"""
Adapted from 5_get_processed_frames.py for a single image.
"""

import os
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from imageio import get_reader, imwrite
import pandas as pd

from utils.config import raw_folder, images_folder, info_folder
from utils.conversion_utils import crop_and_resize

output_dir = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/images/processed_frames/clip_20.png"

corner_points_path  = os.path.join(info_folder, 'corner_points_dictionary.pkl')
points_dict = pd.read_pickle(corner_points_path)
print(points_dict)
# points = [points_dict[case][os.path.splitext(os.path.split(clip)[1].split('_')[2])[0]] for clip in clips]

# frame = img_as_ubyte(rgb2gray(frame))
# # crop the frame, add padding to get the desired aspect ratio, then resize the image
# processed_frame, processing = crop_and_resize(frame, points, output_shape=output_shape, apply_mask=apply_mask)
# # save the processed image and add the processing settings to the list
# imwrite(os.path.join(output_dir, os.path.basename(clip).replace('.mp4', f'_{str(i).zfill(3)}{output_type}')), processed_frame)
