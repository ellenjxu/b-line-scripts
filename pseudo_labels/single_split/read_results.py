"""
Get results from .pkl files.
"""

import pandas as pd

results_file = "C:/Users/ellen/Documents/code/B-line_detection/BEDLUS-Data/trained_networks/results/0201_pseudo-labels_/results_0.9200_max_test/clip_classification_curves/classification_threshold_results.pkl"

results_dict = pd.read_pickle(results_file)
# print(results_dict['result'])

# print keys of results_dict
print(results_dict.keys())