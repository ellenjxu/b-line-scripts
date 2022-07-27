"""
Create split_dictionary_fold.csv for 5-fold cross validation.

Output: split_dictionary_fold.csv
"""

import os
import pandas as pd

info_folder = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/info"
input = os.path.join(info_folder, 'split_dictionary.csv')
output = os.path.join(info_folder, 'split_dictionary_fold.csv')

df = pd.read_csv(input)
train_df = df.loc[df['split'] == 'train']
train_cases = train_df['Case'].unique().tolist()

fold_len = len(train_cases) // 5

for i in range(5):
    fold_cases = train_cases[i*fold_len:(i+1)*fold_len]
    df.loc[df['Case'].isin(fold_cases) & (df['split'] == 'train'), 'split'] = str(i)

# save output .csv
df.to_csv(output, index=False)