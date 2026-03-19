#!/bin/bash
# Run all hyperparameter search experiments for EEGNet-CCA and DNN-CCA.
python run_dnn_cross_experiments_optuna.py
python run_eegnet_experiments_optuna.py
python run_dnncca_cross_experiments_optuna.py
python run_eegnetcca_cross_experiments_optuna.py