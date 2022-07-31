"""
Generate predictions for AI on the training set for crowd.
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
model_name = '0201_pseudo-labels_'
dataset_split = '0' # test, val
extension = ''

# define the detection settings
classification_threshold = 0.920  # for B-line presence in videos, was determined based on validation set result
aggregation_method = 'max'

# define saving settings
store_labels = True

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
split_df = pd.read_csv(os.path.join(info_folder, 'split_dictionary_crowd.csv'))
# drop labels
split_df = split_df.drop(columns=['label'])

# define the directory to the model
model_dir = models_folder if model_subfolder == None else os.path.join(models_folder, model_subfolder)

# loop over all models
# dirs = natsorted(os.listdir(os.path.join(model_dir, model_name)))
# settings_names = [d for d in dirs if d.startswith('experiment_settings') and d.endswith('.pkl')]
# names = [d for d in dirs if d.startswith('final_model') and d.endswith('.pth')] # 'model'

cases = split_df.loc[split_df['split'] == "0"]['Case'].to_list()
clips = split_df.loc[split_df['split'] == "0"]['Clip'].to_list()
print(f'Generating predictions for {model_name} on the {len(cases)} in data split {dataset_split}')

# create output for pseudo labels
pseudo_labels_df = pd.DataFrame(columns=['Case','Clip','pred','label'])

# loop over all cases and clips in the selected split
for case, clip in tqdm(zip(cases, clips)):

    # print(case)

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
    # print(paths)
    # print(clip)

    # get all paths to images that belong to te current clip
    image_paths = [path for path in paths if int(path.split('_')[1]) == int(clip)] # [2]
    # print(image_paths)

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
            
            # replace None by empty tensor
            if y_pred_all == None:
                y_pred_all = torch.zeros((dataset.__len__(), *y_pred.shape[1:]))

            # add the predictions to the storage variable
            last = first + X.shape[0]
            y_pred_all[first:last, ...] += y_pred
            first = last
            
            # set label to 1 if the prediction is above the threshold
            label = [1 if y_pred[i, 1] > classification_threshold else 0 for i in range(y_pred.shape[0])]
            # label = [1 if y_pred > classification_threshold else 0 for y_pred in y_pred]

    # obtain the average by dividing the summed model predictions by the number of models
    # y_pred_all /= len(names)
    
    predictions = y_pred_all[:, 1]
    agg_prediction = aggregate_predictions(predictions, aggregation_method=aggregation_method)
    clip_prediction = 1 if agg_prediction >= classification_threshold else 0
    
    # add prediction to pseudo label df
    pseudo_labels_df = pseudo_labels_df.append({'Case': case, 'Clip': clip, 'pred': y_pred.numpy().tolist(), 'label': clip_prediction}, ignore_index=True)
    
# create an Excel file with pseudo labels
if store_labels:
    pseudo_labels_df.to_csv(os.path.join(info_folder, 'all_ai_labels.csv'), index=False)