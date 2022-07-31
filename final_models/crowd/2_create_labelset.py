"""
Create labelsets for data that was prepared in 1_create_dataset.py.

Output: image_label_combinations.csv
"""

import os
import pandas as pd
from utils.config import images_folder, info_folder

# define name for csv file with image-label combinations
image_label_combinations = 'image_label_combinations.csv'

# define output folder and data type for images
input_folder = os.path.join(images_folder, 'datasets', 'frames_16')
output_folder = os.path.join(images_folder, 'labelsets', 'frames_16')

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# read the csv file with classification results
split_df = pd.read_csv(os.path.join(info_folder, 'split_dictionary_crowd.csv'))

# create a dictionary to keep track of paths to images and corresponding labels
data_dict = {'split': [], 'b-lines_present':[], 'image_path': [], 'label_info': []}

# loop over the dataset splits
for split in ["0", "test"]:
    
    print(f'Split: {split}')

    for classification, label in zip(['pos', 'neg'], [1, 0]):
        # define path to images with a specific label directory and get all filenames
        frames_path = os.path.join(input_folder, split, classification)
        frames = os.listdir(frames_path)

        # update data dictionary
        data_dict['split'] += [split]*len(frames)
        data_dict['b-lines_present'] += [classification]*len(frames)
        data_dict['image_path'] += [os.path.join(frames_path, frame)[len(images_folder)+1:] for frame in frames]
        data_dict['label_info'] += [label]*len(frames)

# save the data dictionary as csv file
data_df = pd.DataFrame(data_dict, columns=['split', 'b-lines_present', 'image_path', 'label_info'])
data_df.to_csv(os.path.join(output_folder, image_label_combinations), index=False)
