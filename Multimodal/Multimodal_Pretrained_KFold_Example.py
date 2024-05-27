#!/usr/bin/env python
# coding: utf-8

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

# Custom functions
from msi_utils_Multimodal import *
from fold_utils_Multimodal import * 
from multimodal_utils import *
from multimodal_model import *

if __name__ == "__main__":
	#Training/Val Set
	df_train_val = pd.read_csv('/path/Train_Val.csv')
	kfold_preds = pd.DataFrame(columns=['target_yield'])

	split_list = kfold_splitter(df=df_train_val)

	# Mixed model variables
	# Set weights for each loss
	tab_w, vis_w, tv_w = 0.42, 0.34, 0.24

	# Initialise Loss
	gb_loss = GradientBlending(tab_weight=tab_w, visual_weight=vis_w, tab_vis_weight=tv_w, loss_scale=1.0)

	# METRICS
	metrics = [t_rmse, v_rmse, tv_rmse, weighted_RMSEp]
	csvlogger = CSVLogger(f'/path/metrics.csv', append=True)
	cbs = [csvlogger]   

	procs = [Categorify, Normalize, FillMissing]
	cont_vars = df_train_val.columns[21:].tolist()
	additional_cont_vars = ['JulianPlantDatePerYear', 'Year', 'DTA', 'DTS', 'Moist', 'Population', 'Range', 'Row']
	cont_names =  cont_vars + additional_cont_vars 
	cat_names = ['Pedigree1', 'Pedigree2', 'Stock', 'Test']

	for i in range(5):
		#Training/Val Path
		path = Path('/path/train_images')
		getter = get_fold(split_list, fold=i)
		splits = getter(range_of(df_train_val))

		to = TabularPandas(df_train_val,
		                   procs,
		                   cat_names=cat_names,
		                   cont_names=cont_names,
		                   y_names='Yield',
		                   y_block=RegressionBlock(),
		                   splits=splits)

		tab_dl = to.dataloaders(bs=64)

		dblock = DataBlock(blocks=(ImageBlock, RegressionBlock),
		            get_items=get_image_files_from_df,
		            get_y=get_y,
		            splitter=getter,
		            item_tfms=[FlipItem, Resize(360, None)],
		            batch_tfms=[Normalize])

		msi_dls = dblock.dataloaders(path, bs=64)

		# Now mix the tabular and spectral datasets to create the multimodal input
		train_mixed_dl = MixedDL(tab_dl[0], msi_dls[0])
		valid_mixed_dl = MixedDL(tab_dl[1], msi_dls[1])
		mixed_dls = DataLoaders(train_mixed_dl, valid_mixed_dl).cuda()
 

		# Modules
		config = tabular_config(ps=0.5, embed_p=0.5)
		learn_tab = tabular_learner(tab_dl,
		                            config=config,
		                            layers=[200,100],
		                            metrics=[rmse, R2Score()],
		                            opt_func=ranger,
		                            y_range=[0,20],
		                            wd=1.425482107813348e-06)

		learn_tab.load('/path/Tabular_Model')

		learn_tab.fit_one_cycle(1, lr_max=0.00018479913871295546)

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
	
		learn_rgb.load('/path/Image_Model')

		learn_rgb.fit(1, lr=0.0001289, wd=0.000137)

		multi_model = TabVis(learn_tab.model, learn_rgb.model)
		multi_learn = Learner(mixed_dls, multi_model, gb_loss, cbs=cbs, metrics=metrics)

		# Disable Fastai progress bar
		with multi_learn.no_bar() and multi_learn.no_logging():
		    multi_learn.fit_one_cycle(35, lr_max=0.000183557)

		pn = msi_dls.valid_ds.items
		images_id = []

		for i in range(len(pn)):
		    path = Path(pn[i])  # Convert the file path to a Path object
		    name = path.stem
		    images_id.append(name)

		preds,targs = multi_learn.get_preds(dl=valid_mixed_dl)
		pred_mixed_df = pd.DataFrame()
		tab_pred = preds[0].flatten()
		vis_pred = preds[1].flatten()
		mixed_pred = preds[2].flatten()

		pred_mixed_df['items'] = images_id
		pred_mixed_df['items'] = pred_mixed_df['items'].str.replace('id_', '')
		pred_mixed_df['tab_pred'] = tab_pred
		pred_mixed_df['msi_pred'] = vis_pred
		pred_mixed_df['mixed_pred'] = mixed_pred
		pred_mixed_df['target_yield'] = targs
		kfold_preds = kfold_preds.append(pred_mixed_df)


	kfold_preds.to_csv('/path/Multimodal_Predictions_Kfold.csv')









