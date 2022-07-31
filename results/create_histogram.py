"""
Create histogram of IOU results from iou_results_final.txt.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')

file = "C:/Users/ellen/Documents/code/B-line_detection/scripts/results/iou_results_final.txt"

iou = []

for row in open(file):
    row = row.strip().split()
    iou.append(float(row[1]))

# kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40, ec="k")

# plt.hist(iou, **kwargs)
# plt.show()

sns.set_style("darkgrid")
sns.despine()

sns.distplot(iou, label='IOU', bins=40, color="#6076a7")
plt.legend()
plt.title('IOU of Automatic Segmentation vs. Manual Segmentation', fontsize=16, font='Verdana')
plt.xlabel('IOU', fontsize=14, font='Verdana')
plt.xlim(0.5, 1)
plt.ylabel('Frequency', fontsize=14, font='Verdana')
plt.plot([0.85, 0.85], [0, 17], 'k--')
plt.savefig('C:/Users/ellen/Documents/code/B-line_detection/scripts/results/iou_histogram.png')
plt.show()