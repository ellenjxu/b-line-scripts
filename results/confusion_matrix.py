"""
Create confusion matrix for diagraming model results.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# cf_matrix = np.array([[95, 16], [11, 94]])
# cf_matrix2 = np.array([[97, 14], [11, 94]])
# cf_matrix3 = np.array([[99, 12], [11, 93]])

cf_matrix = np.array([[120, 17], [9, 52]])
cf_matrix2 = np.array([[135, 2], [13, 48]])
cf_matrix3 = np.array([[136, 1], [16, 45]])

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

# plot all 3 confusion matrices
# divide into 3 subplots
fig, ax = plt.subplots(1, 3, figsize=(14,4))
sns.heatmap(cf_matrix, annot=labels, fmt='', ax=ax[0], cmap='Blues')
sns.heatmap(cf_matrix2, annot=labels, fmt='', ax=ax[1], cmap='Blues')
sns.heatmap(cf_matrix3, annot=labels, fmt='', ax=ax[2], cmap='Blues')

# label each plot
ax[0].set_title("Expert")
ax[1].set_title("Crowd")
ax[2].set_title("AI")

ax[0].set_ylabel("True Labels")
ax[0].set_xlabel("Predicted Labels")

ax[1].set_ylabel("True Labels")
ax[1].set_xlabel("Predicted Labels")

ax[2].set_ylabel("True Labels")
ax[2].set_xlabel("Predicted Labels")

plt.savefig("confusion_matrix.png")
plt.show()