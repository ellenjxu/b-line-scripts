"""
Creates frames_dictionary.pkl with all ~200 clips of the dataset.
"""

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import pickle
import cv2
from glob import glob
from tqdm import tqdm
from natsort import natsorted

from utils.config import raw_folder, info_folder

# define directories, paths, and filenames
input_folder = os.path.join(raw_folder)
frame_dict_name = 'frames_dictionary.pkl'

def get_num_frames(clip: str) -> int:
    """
    Args:
        clip:  path to clip.
    Returns:
        num_frames:  number of frames in a clip.
    """
    cap = cv2.VideoCapture(clip)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return num_frames

# --------------------------------------------------------------------------------

# create dictionaries to collect shape and number of frames of each clip per case
frame_cases = {}

# get all case names from the input folder
cases = sorted([case for case in os.listdir(input_folder) if case.lower().startswith('case')])

for case in tqdm(cases):
    frame_clips = {}
    
    for clip in natsorted(glob(os.path.join(input_folder, case, '*.mp4'))):
        num_frames = get_num_frames(clip)
        clip_name = os.path.splitext(os.path.basename(clip).split('_')[1])[0].zfill(3)
        frame_clips[clip_name] = num_frames

    frame_cases[case] = frame_clips

print(frame_cases)

# save number of frames dictionary variable
file = open(os.path.join(info_folder, frame_dict_name), 'wb')
pickle.dump(frame_cases, file)
file.close()