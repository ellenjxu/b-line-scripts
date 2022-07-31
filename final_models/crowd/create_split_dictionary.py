"""
Create split_dictionary_crowd.csv from all_crowd_labels.csv
"""

import pandas as pd

input_file = "C:/Users/ellen/Documents/code/B-line_detection/scripts/results/all_crowd_labels.csv"
output_file = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/info/split_dictionary_crowd.csv"
split_file = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/info/split_dictionary.csv"

crowd_df = pd.read_csv(input_file)
crowd_df = crowd_df[["Case", "Clip", "label", "split"]]
# print(crowd_df.shape)
split_df = pd.read_csv(split_file)

# only train with patients from training set in split_df
split_df_train = split_df[split_df['split'] == "train"]
crowd_df = crowd_df[crowd_df['Case'].isin(split_df_train['Case'])]
print(crowd_df.shape)

# append all test cases from split_df to crowd_df
split_df_test = split_df[split_df['split'] == "test"]
crowd_df = crowd_df.append(split_df_test)

# print how many unique cases
print(crowd_df['Case'].nunique())

# save to csv
# crowd_df.to_csv(output_file, index=False)