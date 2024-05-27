# Import libraries
from fastai.vision.all import *
import fastai
from fastai.tabular.all import *
from fastai.data.load import _FakeLoader, _loaders
import torch
from ipywidgets import IntProgress
from glob import glob

import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import os

import fastcore

import optuna
from optuna import *

# Custom functions
from msi_utils_Multimodal import *
from fold_utils_Multimodal import * 
from multimodal_utils import *
from multimodal_model import *


# Define a callback function to monitor the process
def callback(study, trial):
	print(f"Trial {trial.number}: Value={trial.value}, Best value={study.best_value}")

def objective_spectral(trial: optuna.Trial):
    # Suggest a learning rate within the specified range [1e-4, 1e-2]
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    
    # Sample weight for tabular model
    tab_w = trial.suggest_float('tab_w', 0.1, 0.8)

    # Calculate weight for visual model based on tab_w
    max_vis_w = min(tab_w, 1.0 - tab_w - 0.1)
    vis_w = trial.suggest_float('vis_w', 0.1, max_vis_w)
    tv_w = 1.0 - tab_w - vis_w  # Weight for combined model

    # Display the selected weights (for demonstration purposes)
    print(f"tab_w: {tab_w}, vis_w: {vis_w}, tv_w: {tv_w}")
    
    #Suggest a batch size
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    

    ##Main body of code including dataloaders and learners##
    # Random splitter function from fastai
    splitter = RandomSplitter(valid_pct=0.3, seed=42)
    splits = splitter(range_of(df_train_val))
    splits

    
    #Tabular Dataloader
    procs = [Categorify, Normalize, FillMissing]
    cont_vars = df_train_val.columns[21:].tolist()
    additional_cont_vars = ['JulianPlantDatePerYear', 'Year', 'DTA', 'DTS', 'Moist', 'Population', 'Range', 'Row']
    cont_names =  cont_vars + additional_cont_vars 
    cat_names = ['Pedigree1', 'Pedigree2', 'Stock', 'Test']

    to = TabularPandas(df_train_val,
                       procs,
                       cat_names=cat_names,
                       cont_names=cont_names,
                       y_names='Yield',
                       y_block=RegressionBlock(),
                       splits=splits)

    tab_dl = to.dataloaders(bs=batch_size)
    

    #Image Dataloader
    dblock = DataBlock(blocks=(ImageBlock, RegressionBlock),
                get_items=get_image_files_from_df,
                get_y=get_y,
                splitter=splitter,
                item_tfms=[FlipItem, Resize(360, None)],
                batch_tfms=[Normalize])

    msi_dls = dblock.dataloaders(path, bs=batch_size)

    #Mixed Dataloader
    # Now mix the tabular and spectral datasets to create the multimodal input
    train_mixed_dl = MixedDL(tab_dl[0], msi_dls[0])
    valid_mixed_dl = MixedDL(tab_dl[1], msi_dls[1])
    mixed_dls = DataLoaders(train_mixed_dl, valid_mixed_dl).cuda()

    # Mixed model variables
    # Initialise Loss
    gb_loss = GradientBlending(tab_weight=tab_w, visual_weight=vis_w, tab_vis_weight=tv_w, loss_scale=1.0)

    # METRICS
    metrics = [t_rmse, v_rmse, tv_rmse, weighted_RMSEp]

    #Tabular Learner
    config = tabular_config(ps=0.5, embed_p=0.5)
    learn_tab = tabular_learner(tab_dl,
                                config=config,
                                layers=[200,100],
                                metrics=[rmse, R2Score()],
                                opt_func=ranger,
                                y_range=[0,20],
                                wd=1.425482107813348e-06)

    learn_tab.fit_one_cycle(1, lr_max=0.00018479913871295546)
    
    #Image Learner
    model_msi = models.densenet121(pretrained=True)

    # Modify the architecture to have 1 output classes
    num_classes = 1
    model_msi.classifier = nn.Linear(model_msi.classifier.in_features, num_classes)

    # Add this line after creating the model architecture
    learn_rgb = Learner(msi_dls,
                model_msi,
                opt_func=RAdam,
                loss_func=root_mean_squared_error,  
                metrics=[rmse, R2Score()])

    learn_rgb.fit(1, lr=0.0001289, wd=0.000137)
    
    #Multimodal Learner
    multi_model = TabVis(learn_tab.model, learn_rgb.model)
    multi_learn = Learner(mixed_dls, multi_model, gb_loss, metrics=metrics)
    
    # Disable Fastai progress bar
    with multi_learn.no_bar() and multi_learn.no_logging():
        multi_learn.fit_one_cycle(35, lr_max=lr)

    if trial.should_prune():
        raise optuna.TrialPruned()

    # returns the [val loss], Optuna focuses on this value.
    return multi_learn.validate()[1]


if __name__ == "__main__":
	# Path to where the images are located
	path = Path('/path/images')
	
	# Load the dataset
	df_train_val = pd.read_csv('/path/Train_Val.csv')

	pruner = optuna.pruners.MedianPruner()
	optuna.logging.set_verbosity(optuna.logging.ERROR)

	study = optuna.create_study(direction="minimize", pruner=pruner, study_name="multimodal_optuna")
	study.optimize(objective_spectral, n_trials=30, callbacks=[callback])

	print("Number of finished trials: {}".format(len(study.trials)))
	print("Best trial:")
	trial = study.best_trial
	print("  Value: {}".format(trial.value))
	print("  Params: ")
	for key, value in trial.params.items():
    		print("    {}: {}".format(key, value))

	# Save trials and study to separate pickle files
	with open('/path/study_trials.pkl', 'wb') as trials_file:
    		pickle.dump(study.get_trials(), trials_file)

	with open('/path/study.pkl', 'wb') as study_file:
    		pickle.dump(study, study_file)