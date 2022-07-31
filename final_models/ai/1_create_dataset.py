"""
Generate data for test prediction set for AI to label based on test_dictionary.csv.
"""

import os
import sys

sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))

import pandas as pd
from natsort import natsorted
import concurrent.futures
from utils.config import images_folder, info_folder, annotations_folder
from utils.dataset_utils import create_datapoint_2, get_all_frames, get_all_frames_from_clip

# specify settings
frames = 16

# define directories to use 
input_folder = os.path.join(images_folder, 'processed_frames')
output_folder = os.path.join(images_folder, 'datasets')

# create output folder with given name if it does not exist yet
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

if not os.path.isdir(os.path.join(output_folder, 'frames_16')):
    os.mkdir(os.path.join(output_folder, 'frames_16'))
else:
    print('frames_16 folder already exists')

if not os.path.isdir(os.path.join(output_folder, 'frames_16', 'test')):
    os.mkdir(os.path.join(output_folder, 'frames_16', 'test'))
else:
    print('test folder already exists')

# read the csv file with classification results
split_df = pd.read_csv(os.path.join(info_folder, 'test_dictionary_ai.csv'))
frames_dict = pd.read_pickle(os.path.join(info_folder, 'frames_dictionary.pkl'))

# loop over the dataset splits
split = "test"
print(f'Split: {split}')

# get all cases in the current split
cases = split_df.loc[split_df['split'] == split]

all_frames = []

for row in cases.itertuples():
    case = row.Case
    # get all frames for the current case
    # all_positive_frames += natsorted(get_all_frames_from_clip(pos_cases[i], pos_cases_clips[i], input_folder))
    for file_path in natsorted(os.listdir(os.path.join(input_folder, case))):
        # print("case in filename", os.path.basename(file_path).split('_')[1])
        # print("case in csv", pos_cases_clips[i])
        if int(os.path.basename(file_path).split('_')[1]) == int(row.Clip):
            all_frames.append(os.path.join(input_folder, case, file_path))

print(f'Copying frames for split {split} to the dataset directory...')

# set some inputs of the convert_clip function
output_directory = os.path.join(output_folder, f'frames_{frames}', split)

if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

adjacent_frames = (0, frames-1)
outside_clip = 'pass'
for i in range(len(all_frames)):
    create_datapoint_2(all_frames[i], output_directory, '.tiff', adjacent_frames, frames_dict, outside_clip) 
