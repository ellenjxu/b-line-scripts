"""
UNUSED
Aggregate the labels to single label in all_ai_labels.csv
"""

import ast
import pandas as pd

df = pd.read_csv("C:/Users/ellen/Documents/code/B-line_detection/scripts/final_models/ai/ai_preds.csv")

# create df for final ai label
df_final = pd.DataFrame(columns=['Case', 'Clip', 'label'])

# TODO: for the same case and clip, get the most common label
for i in range(len(df)):
    if df.iloc[i]['Case'] == df.iloc[i-1]['Case'] and df.iloc[i]['Clip'] == df.iloc[i-1]['Clip']:
        labels = ast.literal_eval(df.iloc[i-1]['labels'])
        labels.append(ast.literal_eval(df.iloc[i]['labels']))

# append labels from each case and clip

# current = [None, None]

# for row in df.itertuples():
#     # if new case and clip is not equal to current case and clip, start new preds list
#     if row.Case != current[0] and row.Clip != current[1]:
#         preds = ast.literal_eval(row.labels)
#         current = [row.Case, row.Clip]
#     elif row.Case == current[0] and row.Clip == current[1]:
#         preds.append(ast.literal_eval(row.labels))
        