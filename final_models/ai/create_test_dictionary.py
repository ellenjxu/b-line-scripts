"""
Get AI pseudo labels for all images in training dataset.

Code:
- get all training images from split_images_crowd (so that the AI annotates the same ones as the crowd); basically the '0' in the images/dataset
- get predictions from the model from pseudo labels "0201_pseudo-labels_" on these images by setting them to "test" set
- output into all_labels_ai.csv
"""

import os
import pandas as pd

crowd_df = pd.read_csv("C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/info/split_dictionary_crowd.csv")
crowd_df = crowd_df.drop(columns=['label'])

# create dictionary for ai test set
ai_df = crowd_df[crowd_df['split'] == "test"]

ai_df.to_csv("C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/info/test_dictionary_ai.csv", index=False)
