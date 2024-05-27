#!/usr/bin/env python
# coding: utf-8

# Import libraries
from fastai.tabular.all import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
import argparse
import optuna
from optuna.integration import FastAIPruningCallback
import fastai.optimizer as optim


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
    
    learn_tab = tabular_learner(data_init,
                            config=config,
                            layers=[200,100],
                            metrics=[rmse, R2Score()],
                            opt_func=optimizer,
                            y_range=[0,20],
                            wd=weight_decay)
    
    # Disable Fastai progress bar and logging
    with learn_tab.no_bar() and learn_tab.no_logging():
        learn_tab.fit_one_cycle(100, lr_max=lr)
    
    if trial.should_prune():
        raise optuna.TrialPruned()

    # returns the [val loss, rmse, r2_score]
    return learn_tab.validate()[1]

if __name__ == "__main__":
	## Load the train/val datasets for DNN
	df_train_val = pd.read_csv('/path/Training_Val.csv')

    # Random splitter function from fastai
    splitter = RandomSplitter(valid_pct=0.3, seed=42)
    splits = splitter(range_of(df_train_val))
    splits

	procs = [Categorify, Normalize, FillMissing]
    
	cont_vars = df_train_val.columns[21:].tolist()
    additional_cont_vars = ['JulianPlantDatePerYear', 'Year', 'DTA', 'DTS', 'Moist', 'Population', 'Range', 'Row']
    cont_names =  cont_vars + additional_cont_vars 
    cat_names = ['Pedigree1', 'Pedigree2', 'Stock', 'Test']

    init = TabularPandas(df_train_val,
                       procs,
                       cat_names=cat_names,
                       cont_names=cont_names,
                       y_names='Yield',
                       y_block=RegressionBlock(),
                       splits=splits)
    
    data_init = init.dataloaders(bs=64)

	config = tabular_config(ps=0.5, embed_p=0.5)

	pruner = optuna.pruners.MedianPruner()
	optuna.logging.set_verbosity(optuna.logging.ERROR)

	study = optuna.create_study(direction="minimize", pruner=pruner, study_name="Tabular_Optuna")
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