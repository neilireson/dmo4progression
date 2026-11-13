#####################################################
# Experimental Parameters
#
#####################################################
# TODO this should be a config object that is saved with the results

import os
import random

from dotenv import load_dotenv
from pathlib import Path

# get variables from .env, e.g. DATA_DIR
load_dotenv()

data_version = os.getenv('DATA_VERSION', 'V1.0.0')
data_dir = os.getenv('DATA_DIR', '../Dataset/')
data_dir = data_dir + '/' + data_version + '/'
config_path = Path("../config/" + data_version)

if data_version == 'V7.3':
    data_file = 'MS_dataset_v.7.3.csv'
    dmo_data_dir_suffix = ' Aggregated DMO Data_V7.3'
    dmo_data_file_suffix = '-weekly_agg_all-21-01-2026.csv'
    participant_id_var = 'local.participant'
    dmo_participant_id_var = 'participant_id'
    dmo_visit_var = 'visit_type'
elif data_version == 'V1.0.0': # first public release from Zenodo
    data_file = 'MS_dataset.csv'
    dmo_data_dir_suffix = '_Aggregated_DMO_Data'
    dmo_data_file_suffix = '-weekly_agg_all.csv'
    participant_id_var = 'participantid'
    dmo_participant_id_var = 'participantid'
    dmo_visit_var = 'visit_number'
else:
    raise ValueError('Data version not supported:', data_version)

def min_increase_for_progression(edss_score):
    return 1.0 if edss_score <= 5.5 else 0.5


# whether to choose the 'first', 'last' or 'random' valid progression or non-progression time point for a participant
progression_select = 'first'

# both recall_macro and balanced_accuracy are defined as the average of recall obtained on each class.
scoring = 'recall_macro'

# how progression is calculated
# Constraint that the minimum previous EDSS score is 3.0, given that was the minimum used for recruitment.
edss_group_scores = {
    'G0': [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
    'G1': [3.0, 3.5, 4.0],
    'G2': [4.5, 5.0, 5.5],
    'G3': [6.0, 6.5, 7.0]
}
edss_group_names = edss_group_scores.keys()
edss_group_type = 'G3'

min_progression_points = 2  # needs to be min 2 to have confirmed progression
# row type is one of: 'participant', 'progression', 'transition'
# 'progression' was an experiment to see if it would produce something interesting, which it doesn't seem to do
data_row_type = 'participant'

drop_DMO_features = False
drop_non_DMO_features = False
drop_features_as_predictors = ['edfscr1l']

# This is the order in which the models will be reported
model_names_ordered = ['SVM', 'MLP', 'KNN', 'LR', 'DT', 'RF', 'XGB']
# List of ML model types to include in the experiment, in the order they are processed
model_types = ['SVM', 'KNN', 'LR', 'DT', 'RF', 'XGB', 'MLP']
# model_types = ['KNN', 'DT']

# default values for cross validation
n_splits = 5
n_repeats = 20

# Feature Selection (FS) parameters
feature_selection_type = 'sfs2'  # should be be sfs2, can be 'fixed' if already generated
show_feature_analysis = False
show_mi_feature_selection = False
show_cluster_feature_selection = False
show_pca_feature_selection = False
show_rfe_feature_selection = False
show_sfs_feature_selection = False
show_feature_agglomeration = False
show_redundant_features = False
# number of parallel jobs to run
# if n_jobs <0, the actual n_jobs is 1 + (n_cpus + n_jobs)
fs_n_jobs = -2

# Mutual Information minimum selection threshold
mi_threshold = 0.0

# Sequential Feature Selection (SFS) features
# see https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
sfs_verbose = 1
sfs_forward = True
sfs_floating = True
# If features are set to <1 then it will use 1 for min and the number of features for max
# Setting max to less than number of features reduces compute time, but if floating is true max should be greater than the desired number of features
sfs_min_features = 1
sfs_max_features = 15  #-1
# SFS with Hyper-Parameter Tuning
sfs_do_hpt = False
sfs_hpt_n_iter = 1  # set to 2 for BayesSearchCV as higher values produce (erratic) worse results
sfs_hpt_verbose = 0
sfs_hpt_n_jobs = 5
# sfs hpt values for cross validation
sfs_hpt_n_repeats = 1

# SFS2 does sfs with multiple restarts
# The parameters allow you to reduce the number of repeats and max_k, to speed up processing.
sfs2_verbose = 0
sfs2_restarts = 20
sfs2_repeats = 5
sfs2_max_k = 10

# under/over sampling
do_data_sampling = False

random_state = 42
random.seed(random_state)

# hyper parameter tuning values
do_hpt = False
model_hpt_type = {
    'SVM': 'grid',
    'MLP': 'bayes',
    'KNN': 'grid',
    'LR': 'grid',
    'DT': 'grid',
    'RF': 'grid',
    'XGB': 'bayes'
}

hpt_n_jobs = -2
hpt_n_iter = 20
hpt_verbose = 0

show_data_distributions = False

# TODO save the config and results for each experiment in separate directory
# where to save figures:
fig_save_dir = '../output/all_features/'
os.makedirs(fig_save_dir, exist_ok=True)

