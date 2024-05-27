#!/usr/bin/env python
# coding: utf-8

# Import libraries
from fastai.vision.all import *
import torch
from ipywidgets import IntProgress
from glob import glob
from fastai.vision.augment import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
from pathlib import Path
from functools import partial
from tqdm import tqdm
from fastai.losses import CrossEntropyLossFlat
from fastai import *
from fastai.data.all import *
import optuna
import fastai.optimizer as optim
import joblib
import argparse
from optuna.integration import FastAIPruningCallback
from torchvision import models
from fastai.vision import models as fastai_models

# Custom functions
from msi_utils_Image import *
from kfold_utils_Image import *

# Define a callback function to monitor the process
def callback(study, trial):
	print(f"Trial {trial.number}: Value={trial.value}, Best value={study.best_value}")

def objective_spectral(trial: optuna.Trial):
    # Suggest a learning rate within the specified range [1e-4, 1e-2]
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    
    # Suggest an optimizer name from the given options
    optimizer = trial.suggest_categorical("optimizer_function", ["ranger", "Adam", "RAdam", "QHAdam"])
    
    # Map the optimizer name to the corresponding optimizer class
    optimizer = getattr(optim, optimizer)
    
    # Suggest a weight decay value within the specified range [1e-6, 1e-3]
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    
    # Learner for RGB (new model for each iteration)
    model_rgb = models.densenet121(pretrained=True)
    
    # Modify the architecture to have 3 output classes
    num_classes = 1
    model_rgb.classifier = nn.Linear(model_rgb.classifier.in_features, num_classes)
    
    # Set custom learning rate, weight decay, and activation function
    learn_rgb = Learner(rgb_dl,
                        model_rgb,
                        opt_func=optimizer,
                        loss_func=root_mean_squared_error,  # Use CrossEntropyLoss for classification
                        metrics=[rmse, R2Score()])  # Use accuracy as the evaluation metric
    
    # Disable Fastai progress bar and logging
    with learn_rgb.no_bar() and learn_rgb.no_logging():
        learn_rgb.fit_one_cycle(30, lr_max=lr, wd=weight_decay)
    
    if trial.should_prune():
        raise optuna.TrialPruned()

    # returns the [val loss, rmse, r2_score]
    return learn_rgb.validate()[1]

if __name__ == "__main__":
	# Path to where the images are located
	path = Path('/path/images')
	
	# Load the dataset
	df = pd.read_csv('/path/Training_Val.csv')

	rgb_fold = DataBlock(blocks=(ImageBlock, RegressionBlock),
    					get_items=get_image_files_from_df,
    					get_y=get_y,
    					splitter=RandomSplitter(valid_pct=0.3, seed=42),
    					item_tfms=[FlipItem, Resize(360, None)],
    					batch_tfms=[Normalize])

	rgb_dl = rgb_fold.dataloaders(path, bs=64)

	pruner = optuna.pruners.MedianPruner()
	optuna.logging.set_verbosity(optuna.logging.ERROR)

	study = optuna.create_study(direction="minimize", pruner=pruner, study_name="study_name")
	study.optimize(objective_spectral, n_trials=50, callbacks=[callback])

	print("Number of finished trials: {}".format(len(study.trials)))
	print("Best trial:")
	trial = study.best_trial
	print("  Value: {}".format(trial.value))
	print("  Params: ")
	for key, value in trial.params.items():
    		print("    {}: {}".format(key, value))

	# Save trials and study to separate pickle files
	with open('/path/studytrials.pkl', 'wb') as trials_file:
    		pickle.dump(study.get_trials(), trials_file)

	with open('/path/study.pkl', 'wb') as study_file:
    		pickle.dump(study, study_file)
