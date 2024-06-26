"""
Generate precision recall curves for crowd and expert.
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, PrecisionRecallDisplay, precision_score, recall_score

# load the data
df = pd.read_csv('C:/Users/ellen/Documents/code/B-line_detection/scripts/results/all_labels.csv')
df = df[['chosen_answer', 'crowd_label', 'expert_chosen_answer', 'gs_label']]
# print(df.head())

classification = ['no b-lines', '1 or more discrete b-lines', 'confluent b-lines']

def test_threshold(thresh, y_labels, y_hat_crowd, y_hat_expert):
    y_hat_crowd_thresh = [1 if y_hat_crowd[i] >= thresh else 0 for i in range(len(y_hat_crowd))]
    y_hat_expert_thresh = [1 if y_hat_expert[i] >= thresh else 0 for i in range(len(y_hat_expert))]

    crowd_precision = precision_score(y_labels, y_hat_crowd_thresh)
    crowd_recall = recall_score(y_labels, y_hat_crowd_thresh)

    expert_precision = precision_score(y_labels, y_hat_expert_thresh)
    expert_recall = recall_score(y_labels, y_hat_expert_thresh)

    return (crowd_precision, crowd_recall), (expert_precision, expert_recall)

for classes in classification:

    y_labels = df['gs_label'].values.tolist()
    y_hat_crowd = df['chosen_answer'].values.tolist()
    y_hat_expert = df['expert_chosen_answer'].values.tolist()

    y_labels = [1 if y_labels[i] == f'{classes}' else 0 for i in range(len(y_labels))]
    # print(y_labels)
    df[f'expert_{classes}_gs'] = y_labels

    # --------------------- Predictions for crowd ---------------------

    y_hat_crowd_probs = []

    for i in range(len(y_hat_crowd)):
        list = y_hat_crowd[i].split(',')
        list = [1 if list[j] == f" '{classes}'" or list[j] == f"['{classes}'" or list[j] == f" '{classes}']" else 0 for j in range(len(list))]
        prob = sum(list) / len(list)
        y_hat_crowd_probs.append(prob)

    # print(y_hat_crowd_probs)
    df[f'crowd_score_{classes}'] = y_hat_crowd_probs

    # --------------------- Predictions for expert ---------------------

    # 1. get curve for all experts

    # y_hat_expert_probs = []

    # for i in range(len(y_hat_expert)):
    #     list = y_hat_expert[i].split(',')
    #     list = [1 if list[j] == f" '{classes}'" or list[j] == f"['{classes}'" or list[j] == f" '{classes}']" else 0 for j in range(len(list))]
    #     prob = sum(list) / len(list)
    #     y_hat_expert_probs.append(prob)

    # # print(y_hat_expert_probs)
    # df[f'expert_score_{classes}'] = y_hat_expert_probs

    # 2. get curve for each expert

    y_hat_expert_probs = [[], [], [], [], [], []]
    y_hat_expert_avg = []

    for i in range(len(y_hat_expert)):
        list = y_hat_expert[i].split(',')
        list = [1 if list[j] == f" '{classes}'" or list[j] == f"['{classes}'" or list[j] == f" '{classes}']" else 0 for j in range(len(list))]
        
        # get average probability
        avg = sum(list) / len(list)
        y_hat_expert_avg.append(avg)
        
        for j in range(6): # append to each individual expert's list
            y_hat_expert_probs[j].append(list[j])

    # print(y_hat_expert_probs)
    # print(len(y_hat_expert_probs))

    # df[f'expert_score_{classes}'] = y_hat_expert_probs

    # --------------------- PR curves ---------------------

    # test
    # print(test_threshold(0.5, y_labels, y_hat_crowd_probs, y_hat_expert_probs))

    # precision and recall curve

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

    for i in range(6):
        
        # 1. plot points for each expert
        y = precision_score(y_labels, y_hat_expert_probs[i])
        x = recall_score(y_labels, y_hat_expert_probs[i])

        plt.scatter(x, y)
        # plt.annotate(f"expert {i+1}", (x, y))
        legend.append(f"expert {i+1}")
        plt.legend(legend)

        # 2. plot curve for each expert
        # precision, recall, _ = precision_recall_curve(y_labels, y_hat_expert_probs[i])
        # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot(ax=plt.gca(), name=f"expert {i+1} (AP: {average_precision_score(y_labels, y_hat_expert_probs[i]):.2f})")

    # plot expert average as cross
    y = precision_score(y_labels, y_hat_expert_avg)
    x = recall_score(y_labels, y_hat_expert_avg)
    
    plt.plot(x, y, '+', color='red')

    plt.title(f'Precision-Recall curve for {classes}')
    plt.savefig(f'C:/Users/ellen/Documents/code/B-line_detection/scripts/results/precision_recall_{classes}.png')
    plt.show()

# df.to_csv('C:/Users/ellen/Documents/code/B-line_detection/scripts/results/precision_recall_out.csv', index=False)
# print(df["expert_score_no b-lines"].value_counts())