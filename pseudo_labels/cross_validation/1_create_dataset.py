"""
Generate data and folder structure with the corresponding .tiffs based on the split_dictionary.csv for 5 fold cross validation.

*Same function as original 1_create_splits.py and 2_create_dataset.py.

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

# read the csv file with classification results
split_df = pd.read_csv(os.path.join(info_folder, 'split_dictionary_fold.csv'))
class_df = pd.read_csv(os.path.join(annotations_folder, 'B-line_expert_classification.csv'))
frames_dict = pd.read_pickle(os.path.join(info_folder, 'frames_dictionary.pkl'))

# create output folder with given name if it does not exist yet
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

if not os.path.isdir(os.path.join(output_folder, 'frames_16')):
    os.mkdir(os.path.join(output_folder, 'frames_16'))
else:
    print('frames_16 folder already exists')

# loop over the dataset splits
for split in split_df['split'].unique():

    # create output folder
    if not os.path.isdir(os.path.join(output_folder, 'frames_16', split)):
        os.mkdir(os.path.join(output_folder, 'frames_16', split))
    else:
        print(f'{split} folder already exists')
    
    print(f'Split: {split}')

    # get all cases in the current split
    cases = split_df.loc[split_df['split'] == split]

    # get all files (case + patient number) with split label
    pos_cases = cases.loc[cases['label'] == 1]['Case'].to_list()
    pos_cases_clips = cases.loc[cases['label'] == 1]['Clip'].to_list()

    neg_cases = cases.loc[cases['label'] == 0]['Case'].to_list()
    neg_cases_clips = cases.loc[cases['label'] == 0]['Clip'].to_list()

    # --------------- POSITIVE FRAMES ------------------

    # get the frame names for all frames in the positively labeled clips 
    # which are part of the selected cases in the current dataset split
    
    all_positive_frames = []

    for i in range(len(pos_cases)):
        # get all frames for the current case
        # all_positive_frames += natsorted(get_all_frames_from_clip(pos_cases[i], pos_cases_clips[i], input_folder))
        for file_path in natsorted(os.listdir(os.path.join(input_folder, pos_cases[i]))):
            # print("case in filename", os.path.basename(file_path).split('_')[1])
            # print("case in csv", pos_cases_clips[i])
            if int(os.path.basename(file_path).split('_')[1]) == int(pos_cases_clips[i]):
                all_positive_frames.append(os.path.join(input_folder, pos_cases[i], file_path))

    print(f'Copying positive frames for split {split} to the dataset directory...')

    # set some inputs of the convert_clip function
    output_directory = os.path.join(output_folder, f'frames_{frames}', split, 'pos')
    
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    adjacent_frames = (0, frames-1)
    outside_clip = 'pass'
    for i in range(len(all_positive_frames)):
        create_datapoint_2(all_positive_frames[i], output_directory, '.tiff', adjacent_frames, frames_dict, outside_clip) 

    # handle clips using multithreading for speedup
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        # copy the negative frames into their new directory
        # executor.map(create, all_positive_frames)

    # --------------- NEGATIVE FRAMES ------------------

    # get the frame names for negatively labeled clips 
    # that are part of all cases in the current dataset split
    
    all_negative_frames = []

    for i in range(len(neg_cases)):
        for file_path in os.listdir(os.path.join(input_folder, neg_cases[i])):
            if int(os.path.basename(file_path).split('_')[1]) == int(neg_cases_clips[i]):
                all_negative_frames.append(os.path.join(input_folder, neg_cases[i], file_path))
    
    # print(all_negative_frames)
    print(f'Copying negative frames for split {split} to the dataset directory...')

    # set some inputs of the convert_clip function
    output_directory = os.path.join(output_folder, f'frames_{frames}', split, 'neg')
    
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    
    adjacent_frames = (0, frames-1)
    outside_clip = 'pass'
    for i in range(len(all_negative_frames)):
        create_datapoint_2(all_negative_frames[i], output_directory, '.tiff', adjacent_frames, frames_dict, outside_clip) 

    # create = lambda source: create_datapoint_2(source, output_directory, '.tiff', adjacent_frames, frames_dict, outside_clip) 

    # handle clips using multithreading for speedup
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        # copy the negative frames into their new directory
        # executor.map(create, all_negative_frames)
