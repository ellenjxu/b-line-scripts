"""
Evaluates frame-level classification performance and optionally plots visualizations.
"""

import os
import sys
sys.path.append(os.path.join('..', '..'))
sys.path.append(os.path.join(__file__, '..', '..', '..'))

import time
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from sklearn.metrics import auc
from torch.utils.data import DataLoader, SequentialSampler

from utils.config import images_folder, models_folder, annotations_folder, info_folder
from utils.evaluation_utils import classify, get_clip_results, aggregate_predictions, isnumber
from utils.augmentation_utils import ClipDataset
from utils.training_utils import get_prediction

# define model evaluation details
model_subfolder = ''
model_name = '0201_crowd-model_'
dataset_split = 'test' # val
extension = ''

# define the detection settings
classification_threshold = 0.766  # for B-line presence in videos, was determined based on validation set result
aggregation_method = 'max'

# define saving settings
store_labels = True
store_spreadsheet = True
store_curve = True

# other parameters
batch_size = 4
auc_step_size = 0.0005
overlap_fraction = 0.0

# --------------------------------------------------------------------------------------------------

# configure the number of workers and the device
num_workers = 0 if sys.platform == 'win32' else 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device\n')

# load the dataset split information
split_df = pd.read_csv(os.path.join(info_folder, 'split_dictionary.csv'))

# define the directory to the model
model_dir = models_folder if model_subfolder == None else os.path.join(models_folder, model_subfolder)

# loop over all models
# dirs = natsorted(os.listdir(os.path.join(model_dir, model_name)))
# settings_names = [d for d in dirs if d.startswith('experiment_settings') and d.endswith('.pkl')]
# names = [d for d in dirs if d.startswith('final_model') and d.endswith('.pth')] # 'model'

# create a subfolder to store all results
result_dir = os.path.join(model_dir, model_name, f'results_{classification_threshold:0.4f}_{aggregation_method}_{dataset_split}{extension}')
if os.path.exists(os.path.join(result_dir)):
    raise IOError('Results directory already exists.')
else:    
    os.mkdir(result_dir)

# create a dictionary to store results
results_dict = {'case': [], 'clip': [], 'N_frames': [], 'predictions': [], 'agg_prediction': [], 
        'clip_prediction': [], 'label': [], 'result': [], 'total_time': [], 'avg_time': []}

cases = split_df.loc[split_df['split'] == "test"]['Case'].to_list()
print(f'Evaluating {model_name} on the data split {dataset_split}')

# create output for pseudo labels
pseudo_labels_df = pd.DataFrame(columns=['Case','Clip','pred','label'])

# loop over all cases and clips in the selected split
for case in cases:

    print(case)

    # get all clips for one case
    clips = natsorted(list(split_df[split_df['Case'] == case]['Clip']))
    
    for clip in tqdm(clips):
        # get the label and convert the clip number to string
        label = int(split_df[(split_df['Case'] == case) & (split_df['Clip'] == clip)]['label'])
        # clip = str(clip).zfill(3)

        # initialize a variable to store the model predictions
        y_pred_all = None

        # start timing
        start = time.perf_counter()

        # loop over the models
        # for name, settings_name in zip(names, settings_names):
        # load the experiment settings and store some settings in variables for later use
        settings = pd.read_pickle(os.path.join(model_dir, model_name, "experiment_settings.pkl"))
        coordinate_system = settings['label_folder'].split('_')[0]
        frames = int(settings['label_folder'].split('_')[1])
        apply_padding = False
        video_dimension = True

        # get all paths
        directory = os.path.join(images_folder, f'processed_frames', case)
        paths = os.listdir(directory)
        print(paths)
        print(clip)

        # get all paths to images that belong to te current clip
        image_paths = [path for path in paths if int(path.split('_')[1]) == int(clip)] # [2]
        print(image_paths)

        # create the dataset and dataloader object
        dataset = ClipDataset(image_paths, directory, frames, overlap_fraction, apply_padding, settings['pretrained'], video_dimension)
        dataloader = DataLoader(dataset, batch_size, sampler=SequentialSampler(dataset), shuffle=False, pin_memory=True)

        # load the model
        model = torch.load(os.path.join(model_dir, model_name, "final_model.pth"))
        model.eval()

        first = 0
        # loop over batches
        with torch.no_grad():
            for X in dataloader:

                # bring the data to the correct device
                X = X.to(device)
                
                # get the prediction
                y_pred = get_prediction(model(X))
                y_pred = torch.softmax(y_pred, dim=1).to('cpu')

                # add prediction to pseudo label df
                pseudo_labels_df = pseudo_labels_df.append({'Case': case, 'Clip': clip, 'pred': y_pred.numpy().tolist(), 'label': label}, ignore_index=True)

                # replace None by empty tensor
                if y_pred_all == None:
                    y_pred_all = torch.zeros((dataset.__len__(), *y_pred.shape[1:]))

                # add the predictions to the storage variable
                last = first + X.shape[0]
                y_pred_all[first:last, ...] += y_pred
                first = last

    # obtain the average by dividing the summed model predictions by the number of models
    # y_pred_all /= len(names)

    # -------------------------  EVALUATION  -------------------------

    # select the foreground predictions, take the average, and compare it to the classification threshold
    N_frames = y_pred_all.shape[0]
    predictions = y_pred_all[:, 1]
    agg_prediction = aggregate_predictions(predictions, aggregation_method=aggregation_method)
    clip_prediction = 1 if agg_prediction >= classification_threshold else 0
    # get the evaluation time
    total_time = time.perf_counter()-start
    avg_time = total_time/N_frames

    # add the clip results to the results dictionary                
    results_dict['clip'].append(clip)
    results_dict['case'].append(case)
    results_dict['N_frames'].append(N_frames) 
    results_dict['predictions'].append(predictions.tolist())
    results_dict['agg_prediction'].append(agg_prediction) 
    results_dict['clip_prediction'].append(clip_prediction)
    results_dict['label'].append(label)
    results_dict['result'].append(classify(clip_prediction, label))
    results_dict['total_time'].append(total_time)
    results_dict['avg_time'].append(avg_time)  

# get the dataframes with the detection results
summary_df, clip_results_df = get_clip_results(results_dict)

# create an Excel file with pseudo labels
if store_labels:
    pseudo_labels_df.to_excel(os.path.join(result_dir, 'pseudo_labels.xlsx'), index=False)

# create an Excel file with the detection results
if store_spreadsheet:
    with pd.ExcelWriter(os.path.join(result_dir, 'individual_clip_results.xlsx')) as writer:
        summary_df.to_excel(writer, sheet_name='summary_results', index=False)
        clip_results_df.to_excel(writer, sheet_name='clip_results', index=False)
        
if store_curve:
    # create the output directory if it does not exist yet
    output_dir = os.path.join(result_dir, f'clip_classification_curves')
    if not os.path.exists(os.path.join(output_dir)):
        os.mkdir(output_dir)

    # define a variable to keep track of the results for different settings of the classification threshold
    track_summary_results = None

    # copy the results dictionary
    results_roc_dict = results_dict.copy()

    threshold_values = list(np.arange(0, 1+auc_step_size, auc_step_size))
    for threshold in tqdm(threshold_values):
        # get the predictions and results based on different classification threshold values
        results_roc_dict['prediction'] = [1 if avg_detections >= threshold else 0 for avg_detections in results_roc_dict['agg_prediction']]
        results_roc_dict['result'] = [classify(pred, lab) for pred, lab in zip(results_roc_dict['prediction'], results_roc_dict['label'])]
        summary_df, _ = get_clip_results(results_roc_dict)
        summary_df['threshold'] = threshold
        summary_df['abs_diff_recall_specificity'] = abs(summary_df['recall']-summary_df['specificity'])

        # append the results
        if track_summary_results == None:
            track_summary_results = summary_df.to_dict('list')
        else:
            summary_dict = summary_df.to_dict('list')
            for key in track_summary_results.keys():
                track_summary_results[key].append(summary_dict[key][0])

    # calculate the AUC of the ROC curve
    roc_auc_value = auc([1-value for value in track_summary_results['specificity']], track_summary_results['recall'])

    # store summary results as a pickle file and as an Excel file
    file = open(os.path.join(output_dir, f'classification_threshold_results.pkl'), 'wb')
    pickle.dump(track_summary_results, file)
    file.close()

    summary_results_df = pd.DataFrame.from_dict(track_summary_results)
    summary_results_df.to_excel(os.path.join(output_dir, f'classification_threshold_results.xlsx'))

    # find the threshold value for which the sensitivity and specificity are (almost) equal,
    # which is equal to the threshold for which the precision and recall match
    equal_row = None
    for index, row in summary_results_df.iterrows():
        if index == 0:
            equal_row = row
        elif row ['abs_diff_recall_specificity'] < equal_row['abs_diff_recall_specificity']:
            equal_row = row
    
    # write a summary file with the results
    equal_threshold = equal_row['threshold']
    equal_recall = equal_row['recall']
    equal_specificity = equal_row['specificity']
    with open(os.path.join(output_dir, f'summary.txt'), 'w') as f:
        f.write(f'{equal_threshold}\t{equal_recall}\t{equal_specificity}\t{roc_auc_value}')

    # create a figure with the detection performance against one of the detection settings
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    ax[0,0].plot(threshold_values, track_summary_results['total TP'], label='total TP', color='limegreen')
    ax[0,0].plot(threshold_values, track_summary_results['total TN'], label='total TN', color='lightgreen')
    ax[0,0].plot(threshold_values, track_summary_results['total FP'], label='total FP', color='red')
    ax[0,0].plot(threshold_values, track_summary_results['total FN'], label='total FN', color='salmon')
    ax[0,0].set_xlabel('classification threshold')
    ax[0,0].set_ylabel('count')
    ax[0,0].set_ylim(bottom=0)
    ax[0,0].set_xticks(np.arange(0, max(threshold_values)+0.01, 0.5))
    ax[0,0].legend()

    ax[0,1].plot(threshold_values, track_summary_results['precision'], label='precision', color='lightskyblue')
    ax[0,1].plot(threshold_values, track_summary_results['recall'], label='recall', color='royalblue')
    ax[0,1].plot(threshold_values, track_summary_results['f1-score'], label='f1-score', color='midnightblue')
    ax[0,1].plot(threshold_values, track_summary_results['specificity'], label='specificity', color='salmon')
    ax[0,1].plot(threshold_values, track_summary_results['accuracy'], label='accuracy', color='limegreen')
    ax[0,1].set_xlabel('classification threshold')
    ax[0,1].set_ylabel('score')
    ax[0,1].set_xticks(np.arange(0, max(threshold_values)+0.01, 0.5))
    ax[0,1].set_yticks(np.arange(0, 1.01, 0.2))
    ax[0,1].legend()

    ax[1,0].plot(track_summary_results['recall'], track_summary_results['precision'], color='royalblue')
    ax[1,0].set_xlabel('Recall')
    ax[1,0].set_ylabel('Precision')
    ax[1,0].set_xticks(np.arange(0, 1.01, 0.2))
    ax[1,0].set_yticks(np.arange(0, 1.01, 0.2))

    ax[1,1].plot([1-value for value in track_summary_results['specificity']], track_summary_results['recall'], color='royalblue')
    ax[1,1].set_xlabel('False positive rate')
    ax[1,1].set_ylabel('True positive rate')
    ax[1,1].set_xticks(np.arange(0, 1.01, 0.2))
    ax[1,1].set_yticks(np.arange(0, 1.01, 0.2))
    ax[1,1].set_title(f'AUC: {roc_auc_value:0.4f}')

    fig.suptitle(f'Classification threshold: {min(threshold_values):0.2f}-{max(threshold_values):0.2f}')
    plt.savefig(os.path.join(output_dir, f'classification_threshold_results.png'))
    plt.close()
