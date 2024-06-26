"""
Generate precision recall curves for crowd, expert, and AI pseudo labels on the test set, for only No B-lines or Present B-lines.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, PrecisionRecallDisplay, precision_score, recall_score
import ast

# load the data
df = pd.read_csv('C:/Users/ellen/Documents/code/B-line_detection/scripts/results/all_labels.csv')
df = df[['Case', 'Clip', 'chosen_answer', 'crowd_label_train', 'expert_chosen_answer', 'gs_label_train', 'split']]   # crowd pred, crowd label, expert pred, expert label
# only get rows that are in 'test' split
df = df.loc[df['split'] == 'test']

df_pseudo = pd.read_excel('C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/trained_networks/results/0201_pseudo-labels_/results_0.9200_max_test/pseudo_labels.xlsx')
# get rid of duplicate cases and clips
df_pseudo = df_pseudo.drop_duplicates(subset=['Case', 'Clip'])

xls = pd.ExcelFile('C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/trained_networks/results/0201_pseudo-labels_/results_0.9200_max_test/individual_clip_results.xlsx')
df2_pseudo = pd.read_excel(xls, 'clip_results')
df2_pseudo = df2_pseudo.drop_duplicates(subset=['case', 'clip'])

df_baseline = pd.read_excel('C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/trained_networks/results/0201_baseline-test_0/classification_threshold_results.xlsx')
# get rid of duplicate thresholds
df_baseline = df_baseline.drop_duplicates(subset=['threshold'])

# get labels for each group
y_labels = df['gs_label_train'].values.tolist()
y_hat_crowd = df['chosen_answer'].values.tolist()
y_hat_expert = df['expert_chosen_answer'].values.tolist()

df[f'expert_0_gs'] = y_labels

# convert 3 classes to 2 classes in crowd and expert preds
correct = "no b-lines"

# --------------------- Predictions for crowd ---------------------

y_hat_crowd_probs = []
    
for i in range(len(y_hat_crowd)):
    list = y_hat_crowd[i].split(',')
    list = [0 if list[j] == f" '{correct}'" or list[j] == f"['{correct}'" or list[j] == f" '{correct}']" else 1 for j in range(len(list))]
    prob = sum(list) / len(list)
    y_hat_crowd_probs.append(prob)

# print(y_hat_crowd_probs)
df[f'crowd_score_0'] = y_hat_crowd_probs

# --------------------- Predictions for expert ---------------------

y_hat_expert_probs = [[], [], [], [], [], []]

for i in range(len(y_hat_expert)):
    list = y_hat_expert[i].split(',')
    list = [0 if list[j] == f" '{correct}'" or list[j] == f"['{correct}'" or list[j] == f" '{correct}']" else 1 for j in range(len(list))]
    
    for j in range(6): # append to each individual expert's list
        y_hat_expert_probs[j].append(list[j])

# print(y_hat_expert_probs)
# print(len(y_hat_expert_probs))

# df[f'expert_score_0'] = y_hat_expert_probs

# --------------------- Predictions for AI ---------------------

# 1. single clip predictions
# get first value of first prediction in each row

# y_hat_ai_probs = []

# for i in range(len(df_pseudo)):
#     # turn to list
#     pred_list = ast.literal_eval(df_pseudo.iloc[i]['pred'])
#     pred = pred_list[0][1]
#     y_hat_ai_probs.append(pred)

# y_ai_labels = df_pseudo['label'].values.tolist()

# 2. aggregate clip predictions

y_hat_ai_probs = df2_pseudo['agg_prediction'].values.tolist()
y_ai_labels = df2_pseudo['label'].values.tolist()

# --------------------- PR curves ---------------------

# test
# print(test_threshold(0.5, y_labels, y_hat_crowd_probs, y_hat_expert_probs))

# precision and recall curve

# print(y_labels)
# print(y_hat_crowd_probs)

precision, recall, _ = precision_recall_curve(y_labels, y_hat_crowd_probs)

# 1. AP
# ap_crowd = average_precision_score(y_labels, y_hat_crowd_probs)
# ap_expert = average_precision_score(y_labels, y_hat_expert_probs)

# 2. AUC
auc_crowd = auc(recall, precision)

disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot(ax=plt.gca(), name=f"crowd (AUC: {auc_crowd:.2f})")
# precision2, recall2, _ = precision_recall_curve(y_labels, y_hat_expert_probs)
# disp = PrecisionRecallDisplay(precision=precision2, recall=recall2)
# disp.plot(ax=plt.gca(), name=f"expert (AP: {ap_expert:.2f})")

# get precision and recall for each expert

legend = [f"crowd (AUC: {auc_crowd:.2f})"]

expert_prec = []
expert_recall = []

# for i in range(6):
    
#     # 1. plot points for each expert
#     y = precision_score(y_labels, y_hat_expert_probs[i])
#     x = recall_score(y_labels, y_hat_expert_probs[i])

#     plt.scatter(x, y)
#     # plt.annotate(f"expert {i+1}", (x, y))
#     legend.append(f"expert {i+1}")
#     plt.legend(legend)
    
#     expert_prec.append(y)
#     expert_recall.append(x)

    # 2. plot curve for each expert
    # precision, recall, _ = precision_recall_curve(y_labels, y_hat_expert_probs[i])
    # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    # disp.plot(ax=plt.gca(), name=f"expert {i+1} (AP: {average_precision_score(y_labels, y_hat_expert_probs[i]):.2f})")

# print(y_ai_labels)
# print(y_hat_ai_probs)

precision2, recall2, _ = precision_recall_curve(y_ai_labels, y_hat_ai_probs)
auc_ai = auc(recall2, precision2)

disp = PrecisionRecallDisplay(precision=precision2, recall=recall2)
disp.plot(ax=plt.gca(), name=f"ai (AUC: {auc_ai:.2f})")

# second ai curve with dashed line
precision3 = df_baseline['precision'].values.tolist()
recall3 = df_baseline['recall'].values.tolist()

# auc_baseline = auc(recall3, precision3)
auc_baseline = 0.94

# make precision3 and recall3 strictly incre
disp = PrecisionRecallDisplay(precision=precision3, recall=recall3)
disp.plot(ax=plt.gca(), name=f"large ai (AUC: {auc_baseline:.2f})", linestyle='--')

# plot average of experts
# plt.plot(sum(expert_prec)/6, sum(expert_recall)/6, '+', mew=2, ms=7, color="black")
# legend.append(f"average of experts")

plt.title(f'Precision-Recall curve for detecting presence of B-lines')
plt.savefig(f'C:/Users/ellen/Documents/code/B-line_detection/scripts/results/all/precision_recall_all_1.png')
plt.show()

# df.to_csv('C:/Users/ellen/Documents/code/B-line_detection/scripts/results/all/precision_recall_out.csv', index=False)
# print(df["expert_score_no b-lines"].value_counts())