# Generated from: data_preprocessing_and_analysis.ipynb
# Converted at: 2026-11-13T12:25:01.022Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

#####################################################
# imports
#####################################################

import numpy as np
import os
import pandas as pd
import seaborn as sns
import random
import inspect

# Note that suppressing warnings using the "warnings" package does not work when using multiprocessing
# see https://github.com/scikit-learn/scikit-learn/issues/12939
import warnings

warnings.filterwarnings('ignore')

# This works for most things, but obviously removes potentially important warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
# This redirects all output to standard error into a file, which at least makes the output clean
# import sys
# sys.stderr = open('log.txt', 'w')

from collections import defaultdict

from dotenv import load_dotenv

from pathlib import Path

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.metrics import sensitivity_score, specificity_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_rel, t

from sklearn import set_config
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, \
    classification_report, confusion_matrix, silhouette_score, \
    roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, \
    RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.cluster import KMeans

from skopt import BayesSearchCV

from timeit import default_timer as timer

from xgboost import XGBClassifier

import matplotlib.colors as colors
import matplotlib.pyplot as plt

import statsmodels.api as sm
import pauc

# make transform operations return dataframes
set_config(transform_output='pandas')

# remove dtype from printing
np.set_printoptions(legacy='1.25')


#####################################################
# Experimental Parameters
#
#####################################################
# TODO this should be a config object that is saved with the results

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


#####################################################
# Load data
#
#####################################################

# read in main MS dataset
data_file_path = data_dir + data_file
all_data = pd.read_csv(data_file_path)
all_data['visit.number'] = all_data['visit.number'].str.upper()  # for consistency with filenames
# convert all column names to lower case for consistency between versions
all_data.columns = all_data.columns.str.lower()

# get participant demographics
demo_data = all_data.copy()
demo_data = demo_data[demo_data['visit.number'] == 'T1']

# after getting table values, just crop to the necessary columns
cols_from_dataset = [participant_id_var, 'edfscr1l', 'visit.number',
                     'age', 'gender', 'mstypeonset', 'mstypediag', 'visit.date']
data = all_data[cols_from_dataset]

# forward fill within groups to ensure no leakage of demographics between participants
cols_from_dataset = [col for col in cols_from_dataset if col in ['age', 'gender', 'mstypeonset', 'mstypediag']]
for col in cols_from_dataset:
    data[col] = data.groupby(participant_id_var)[col].ffill()

# we need to get the time difference and convert from timedelta to something that the ML models can use - probably unix time
data['time_since_diag'] = (pd.to_datetime(data['visit.date']) - pd.to_datetime(data['mstypediag'])).dt.total_seconds()
data['time_since_onset'] = (pd.to_datetime(data['visit.date']) - pd.to_datetime(data['mstypeonset'])).dt.total_seconds()

# drop cols used for calculation
data = data.drop(['mstypeonset', 'mstypediag', 'visit.date'], axis=1)

# convert gender to 0 and 1
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

# print(f'Column NaN counts out of {len(data)} rows')
# print(data.isna().sum())
print(f'Nan counts at each timepoint')
g = data.groupby('visit.number')
print(g.count().rsub(g.size(), axis=0))

# drop rows with no EDSS score
count_before = data[participant_id_var].nunique()
print(f"Dropping {sum(data['edfscr1l'].isna())} rows with no EDSS score...")
data = data.dropna(subset=['edfscr1l'])
print(
    f'Dropping rows results in {count_before - data[participant_id_var].nunique()} participants excluded with no EDSS scores...')

print(f'Nan counts at each timepoint')
g = data.groupby('visit.number')
print(g.count().rsub(g.size(), axis=0))

# Crop down to just the necessary columns
data = data.sort_values(by=[participant_id_var, 'visit.number'])


def demographics_table(data):
    PDDS_lookup = {
        "Normal": 0,
        "Mild Disability": 1,
        "Moderate Disability": 2,
        "Gait Disability": 3,
        "Early Cane": 4,
        "Late Cane": 5,
        "Bilateral Support": 6,
        "Wheelchair/Scooter": 7,
        "Bedridden": 8
    }

    demographics = {}
    demographics["N"] = len(data)
    demographics["Age, mean (SD)"] = f"{data['age'].mean():.1f} ({data['age'].std():.1f})"
    demographics[
        "Gender, female, n (%)"] = f"{len(data[data['gender'] == 'Female'])} ({100 * (len(data[data['gender'] == 'Female']) / len(data)):.1f})"

    time_diff_days = (pd.to_datetime(data['visit.date']) - pd.to_datetime(data['mstypediag'])).dt.days
    time_diff_years = time_diff_days / 365.25
    avg_years = time_diff_years.median()
    std_years = time_diff_years.quantile(0.75) - time_diff_years.quantile(0.25)
    demographics["Years since diagnosis of MS, median (IQR)"] = f"{avg_years:.1f} ({std_years:.1f})"

    time_diff_days = (pd.to_datetime(data['visit.date']) - pd.to_datetime(data['mstypeonset'])).dt.days
    time_diff_years = time_diff_days / 365.25
    avg_years = time_diff_years.median()
    std_years = time_diff_years.quantile(0.75) - time_diff_years.quantile(0.25)
    demographics["Years since onset of symptoms, median (IQR)"] = f"{avg_years:.1f} ({std_years:.1f})"

    """
    college_higher = len(data[data["edhigh"] == "Undergraduate/Bachelor"]) + len(data[data["edhigh"] == "Postgraduate (Diploma, Master etc)"]) + len(data[data["edhigh"] == "Doctoral/PhD or higher"]) + len(data[(data["edhigh"] == "Other") & (data["edyears"] > 15)])
    high_school = len(data[data["edhigh"] == "High school leaving certificate/diploma or equivalent (11 or more years)"]) + len(data[data["edhigh"] == "Tertiary education (e.g. trade or apprenticeship)"]) + len(data[data["edhigh"] == "College/University but no degree"]) + len(data[(data["edhigh"] == "Other") & (data["edyears"] > 10) & (data["edyears"] <= 15)])
    less_than_high_school = len(data[data["edhigh"] == "Less than high school (0-10 years of schooling)"]) + len(data[(data["edhigh"] == "Other") & (data["edyears"] <= 10)])
    demographics["Education, n (%)"] = {
        "College degree or higher": f"{college_higher} ({100*(college_higher/len(data)):.1f})",
        "High school": f"{high_school} ({100*(high_school/len(data)):.1f})",
        "Less than high school": f"{less_than_high_school} ({100*(less_than_high_school/len(data)):.1f})"
    }
    """
    demographics[
        "EDSS, median (IQR)"] = f"{data['edfscr1l'].median()} ({data['edfscr1l'].quantile(0.75) - data['edfscr1l'].quantile(0.25)})"

    demographics["Primary Progressive MS, n"] = len(data[data["mstype"] == "Primary Progressive MS"])
    demographics["Secondary Progressive MS, n"] = len(data[data["mstype"] == "Secondary Progressive MS"])
    demographics["Relapsing Remitting MSS, n"] = len(data[data["mstype"] == "Relapsing Remitting MS"])

    demographics[
        "Indoor Walking Aid, n (%)"] = f"{len(data[data['aidindoor'] == True])} ({100 * (len(data[data['aidindoor'] == True]) / len(data)):.1f})"
    demographics[
        "Outdoor Walking Aid, n (%)"] = f"{len(data[data['aidoutdoor'] == True])} ({100 * (len(data[data['aidoutdoor'] == True]) / len(data)):.1f})"
    #demographics["Falls in the previous 12 months, mean (SD)"] = f"{data_all["fallsnum"].dropna().astype(int).mean():.1f} ({data_all["fallsnum"].dropna().astype(int).std():.1f})"
    demographics["6MWT, mean (SD)"] = f"{data['6mwd1'].mean():.1f} ({data['6mwd1'].std():.1f})"
    #demographics["Systolic BP, mmHg, mean (SD)"] = f"{data["bpsyst"].mean():.1f} ({data["bpsyst"].std():.1f})"

    demographics["MSWS-12, mean (SD)"] = f"{data['mswtot1n'].mean():.1f} ({data['mswtot1n'].std():.1f})"
    demographics["T25-FW, mean (SD)"] = f"{data['tfwmn1n'].mean():.1f} ({data['tfwmn1n'].std():.1f})"
    demographics[
        "PDDS, median (IQR)"] = f"{data['pddsw1l'].map(PDDS_lookup).median():.1f} ({(data['pddsw1l'].map(PDDS_lookup).quantile(0.75) - data['pddsw1l'].map(PDDS_lookup).quantile(0.25)):.1f})"

    return demographics


def get_characteristic_table(edss_group_participant_data, data):
    data = data[data["visit.number"] == "T1"].reset_index(drop=True)

    combined = []
    for name in edss_group_scores.keys():
        combined.append(demographics_table(
            data[data[participant_id_var]
            .isin(edss_group_participant_data[name][participant_id_var])]))

    combined_df = pd.DataFrame(combined).transpose()
    combined_df.columns = list(edss_group_scores.keys())
    print(combined_df.to_latex())




#####################################################
# Functions to calculate participant-based progression
#
#####################################################

timepoints = ['T1', 'T2', 'T3', 'T4', 'T5']


def calc_progression(tp_scores, min_score, return_all=False, select=progression_select):
    progressions = []
    non_progressions = []

    for i, tp0 in enumerate(timepoints):
        if i == len(tp_scores) - min_progression_points:
            break
        if tp_scores[tp0] >= min_score:
            valid_future_points = []
            for j in range(i + 1, len(timepoints)):
                if not np.isnan(tp_scores.iloc[j]):
                    valid_future_points.append(tp_scores.iloc[j])
            # if there are at least min_progression_points future non-null timepoints, then this is valid
            if len(valid_future_points) >= min_progression_points:
                min_progression = min_increase_for_progression(tp_scores[tp0])
                is_progression = True
                # if any of the consecutive future timepoint scores is
                # less than required for progression, this is not progression
                for j in range(min_progression_points):
                    if valid_future_points[j] - tp_scores[tp0] < min_progression:
                        is_progression = False
                        break

                if is_progression:
                    progressions.append({'timepoint': tp0, 'next_edss': valid_future_points[0], 'progression': 1})
                else:
                    non_progressions.append({'timepoint': tp0, 'next_edss': valid_future_points[0], 'progression': 0})
            else:  # if no valid future timepoints
                break

    if return_all:
        return progressions, non_progressions
    # if there are no valid values
    if len(progressions) == 0 and len(non_progressions) == 0:
        return {'timepoint': None, 'next_edss': None, 'progression': None}

    # return progression if they exist, otherwise non_progression
    result = progressions if len(progressions) > 0 else non_progressions
    if len(result) == 1:
        return result[0]
    match select:
        case 'first':
            return result[0]
        case 'last':
            return result[-1]
        case _:  # random
            return result[random.randint(0, len(result) - 1)]



#####################################################
# Calculate Participant-based Progression
#
#####################################################

# Add the DMO data so that timepoints with missing DMOs are not considered in the calculations
dfs_list = []
for timepoint in ['T1', 'T2', 'T3']:
    feature_file_path = data_dir + f'/{timepoint}{dmo_data_dir_suffix}/cvs-{timepoint}{dmo_data_file_suffix}'
    temp_df = pd.read_csv(feature_file_path)
    dfs_list.append(temp_df)

dmo_data = pd.concat(dfs_list, ignore_index=True)
# convert all column names to lower case for consistency between versions
dmo_data.columns = dmo_data.columns.str.lower()
dmo_data[dmo_visit_var] = dmo_data[dmo_visit_var].str.upper()  # for consistency with filenames

merged_data = pd.merge(data, dmo_data, left_on=[participant_id_var, 'visit.number'],
                       right_on=[dmo_participant_id_var, dmo_visit_var], how='left')

if not dmo_participant_id_var == participant_id_var:
    merged_data = merged_data.drop([dmo_participant_id_var], axis=1)
merged_data = merged_data.drop([dmo_visit_var], axis=1)

# print(f'Column NaN counts out of {len(merged_data)} T1, T2, T3 rows')
# print(merged_data[merged_data['visit.number'].isin({'T1', 'T2', 'T3'})].isna().sum())

# drop T1, T2, and T3 rows if they contain nan, which will be due to missing DMOs
count_before = merged_data[participant_id_var].nunique()
print(f'{count_before} participants read from file...')

# get indices of rows with any nan values
nan_indices = merged_data.index[merged_data.isna().any(axis=1)]
# get indices of rows with T1, T2 or T3 values
timepoint_indices = merged_data.index[merged_data['visit.number'].isin({'T1', 'T2', 'T3'})]
# Get intersection of T1, T2, T3 rows with rows containing any nan
intersection = nan_indices.intersection(timepoint_indices)
merged_data = merged_data.drop(intersection)
print(
    f'{count_before - merged_data[participant_id_var].nunique()} participants excluded with missing DMOs at timepoints T1, T2 and T3.')

participant_data = (merged_data[[participant_id_var, 'visit.number', 'edfscr1l']]
                    .pivot(index=participant_id_var, columns='visit.number', values='edfscr1l'))

count_before = len(participant_data)
participant_data = participant_data[participant_data[['T1', 'T2', 'T3']].isna().sum(axis='columns') < 3]
print(
    f'{count_before - len(participant_data)} participants excluded as have T1-3 EDSS scores are all NaN...')

base_scores = edss_group_scores['G0']

count_before = len(participant_data)
participant_data = participant_data[participant_data[['T1', 'T2', 'T3']].le(base_scores[-1]).any(axis='columns')]
print(
    f'{count_before - len(participant_data)} participants excluded as have T1-3 EDSS scores > threshold ({base_scores[-1]})...')

# Add progression data with baseline timepoint, next_edss score after baseline, and whether baseline is followed by a confirmed progression (0,1)
participant_data[['timepoint', 'next_edss', 'progression']] = participant_data.apply(
    # threshold_progress_point, thresholds=base_scores, min_score=base_scores[0],
    calc_progression, min_score=base_scores[0],
    axis='columns', result_type='expand')

# calculate all progressions data
progression_data = []
for index, row in participant_data.iterrows():
    progressions, non_progressions = calc_progression(row, base_scores[0], return_all=True)
    for progression in progressions:
        progression_data.append([index, progression['timepoint'], progression['next_edss'], 1])
    for non_progression in non_progressions:
        progression_data.append([index, non_progression['timepoint'], non_progression['next_edss'], 0])
progression_data = pd.DataFrame(progression_data,
                                columns=[participant_id_var, 'timepoint', 'next_edss', 'progression'])

print('Participants excluded as they have no valid timepoints')
print(participant_data[participant_data['timepoint'].isna()])
# remove all participants who have insufficient (i.e. <3) timepoints
count_before = len(participant_data)
participant_data = participant_data.dropna(subset=['timepoint'])
print(f'{count_before - len(participant_data)} participants excluded who have insufficient (i.e. <3) timepoints...\n')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(participant_data[participant_data['progression'] == 1].head(100))

participant_data = participant_data.merge(merged_data, how='left',
                                          left_on=[participant_id_var, 'timepoint'],
                                          right_on=[participant_id_var, 'visit.number'])
# remove processing columns
participant_data = participant_data.drop(['T1', 'T2', 'T3', 'T4', 'T5', 'timepoint'], axis=1)

progression_data = progression_data.merge(merged_data, how='left',
                                          left_on=[participant_id_var, 'timepoint'],
                                          right_on=[participant_id_var, 'visit.number'])
# remove processing column
progression_data = progression_data.drop(['timepoint'], axis=1)

edss_group_participant_data = {}
for group in edss_group_scores.keys():
    edss_group_participant_data[group] = participant_data[participant_data['edfscr1l'].isin(edss_group_scores[group])]
    print(f'EDSS group {group} size: {len(edss_group_participant_data[group])}')

get_characteristic_table(edss_group_participant_data, all_data)

if not edss_group_type == 'G0':
    print(f'Selecting EDSS scores for group {edss_group_type}: {edss_group_scores[edss_group_type]}')
    count_before = len(participant_data)
    participant_data = edss_group_participant_data[edss_group_type]
    print(f'Participants data: {count_before - len(participant_data)} participants excluded who are not in the group')

    participants_before = progression_data[participant_id_var].nunique()
    progressions_before = len(progression_data)
    progression_data = progression_data[progression_data['edfscr1l'].isin(edss_group_scores[edss_group_type])]
    print(
        f'Progression data: {participants_before - progression_data[participant_id_var].nunique()} participants excluded who are not in the group')
    print(f'Progression data: {progressions_before - len(progression_data)} progressions excluded from group')

non_progressor_count = len(participant_data[participant_data['progression'] == 0])

print(f'Number of included participants = {len(participant_data)}')
print(f'Number of non-progressors = {non_progressor_count}')
print(f'Number of progressors = {len(participant_data) - non_progressor_count}')

print(pd.crosstab(participant_data['progression'], participant_data['edfscr1l']))

ax = (pd.crosstab(participant_data['edfscr1l'], participant_data['progression'])
      .plot(kind='bar',
            stacked=True,
            color=['skyblue', 'darkblue'],
            alpha=0.8,
            figsize=(10, 6)))

plt.title('Progression by EDSS Score')
plt.xlabel('EDSS Score')
plt.ylabel('Number of Participants')
plt.xticks(rotation=0)
plt.legend(['Non-progressors', 'Progressors'])
plt.savefig('../output/progressors_' + edss_group_type + '.png')
plt.show()

#print(participant_data[participant_id_var].unique())


############################################################
# Calculate EDSS score consecutive timepoint transitions
#
############################################################

def show_transition_matrix(data, name):
    # Create matrix of prev -> next EDSS scores
    prev_scores = data['edfscr1l'].to_numpy().tolist()
    next_scores = data['next_edss'].to_numpy().tolist()

    # extract the unique scores
    scores = list(set(prev_scores).union(set(next_scores)))
    scores.sort()
    n_scores = len(scores)

    # Create transition matrix
    # Note some of these transitions will be for people with missing DMOs
    transition_matrix = pd.DataFrame(
        np.zeros((n_scores, n_scores), dtype=int),
        index=scores,
        columns=scores)

    for prev_score, next_score in zip(prev_scores, next_scores):
        transition_matrix.loc[prev_score, next_score] += 1

    # Pretty print transition matrix as heatmap
    plt.subplots(figsize=(10, 8))
    ax = plt.subplot(111)
    sns.heatmap(transition_matrix.values, annot=True, cmap='Blues', ax=ax, fmt='d', norm=colors.LogNorm())
    ax.set_ylabel('Previous Score')
    ax.set_xlabel('Next Score')
    ax.xaxis.set_ticklabels(scores)
    ax.yaxis.set_ticklabels(scores)
    plt.savefig('../output/' + name + '_' + edss_group_type + '.png')
    plt.show()


transition_data = data.copy()
print(f'{transition_data[participant_id_var].nunique()} participants')
print(f'{len(transition_data)} total participant timepoints rows')

# shift timepoints and EDSS to get next as columns
transition_data['next_edss'] = transition_data.groupby(participant_id_var)['edfscr1l'].shift(-1)
transition_data['Next_Timepoint'] = transition_data.groupby(participant_id_var)['visit.number'].shift(-1)

transition_data = transition_data.dropna(subset=['Next_Timepoint'])

# Data cleaning - will monitor number of rows throughout
print(f'{len(transition_data)} rows after dropping NaN in next timepoint...')

# get rid of participants where either current or next EDSS is missing
count_before = len(transition_data)
transition_data = transition_data.dropna(subset=['edfscr1l', 'next_edss'])
print(f'{count_before - len(transition_data)} rows have missing EDSS values and will be dropped...')

show_transition_matrix(transition_data, 'all_transitions')

# Exclude participants with current EDSS score >= threshold as they have already progressed:
count_before = len(transition_data)
high_prev_edss_mask = (transition_data['edfscr1l'] > base_scores[-1])
print(f'{sum(high_prev_edss_mask)} rows have a EDSS > {base_scores[-1]} and will be dropped...')
transition_data = transition_data.drop(transition_data[high_prev_edss_mask].index)

# Exclude participants with EDSS score < min_prev_score as they are highly unlikely to progress, and may have different mobility patterns
count_before = len(transition_data)
low_prev_edss_mask = (transition_data['edfscr1l'] < base_scores[0])
print(f'{sum(low_prev_edss_mask)} rows have a EDSS < {base_scores[0]} and will be dropped...')
transition_data = transition_data.drop(transition_data[low_prev_edss_mask].index)


def progression(row):
    return 1 if ((row['edfscr1l'] > base_scores[0]) &
                 (row['next_edss'] >= base_scores[0]) &
                 ((row['next_edss'] - row['edfscr1l']) >= min_increase_for_progression(row['edfscr1l']))) else 0


# get progression label column
transition_data['progression'] = transition_data.apply(progression, axis=1)

# transitions = transition_data[progression_mask]  # all the data where a transition happens, just in case.
#print(transitions.head(10))

# now that we have a reduced dataset, loop through our weekly dmo aggregations and append the relevant ones to the dataframe

dfs_list = []
# we don't need DMOs where T5 is previous
for timepoint in ['T1', 'T2', 'T3', 'T4']:
    feature_file_path = data_dir + f'/{timepoint}{dmo_data_dir_suffix}/cvs-{timepoint}{dmo_data_file_suffix}'
    temp_df = pd.read_csv(feature_file_path)
    dfs_list.append(temp_df)

dmo_data = pd.concat(dfs_list, ignore_index=True)
# convert all column names to lower case for consistency between versions
dmo_data.columns = dmo_data.columns.str.lower()
dmo_data[dmo_visit_var] = dmo_data[dmo_visit_var].str.upper()  # for consistency with filenames

transition_data = pd.merge(transition_data, dmo_data, left_on=[participant_id_var, 'visit.number'],
                           right_on=[dmo_participant_id_var, dmo_visit_var], how='left')
if not dmo_participant_id_var == participant_id_var:
    transition_data = transition_data.drop([dmo_participant_id_var], axis=1)
transition_data = transition_data.drop([dmo_visit_var], axis=1)

# drop all remaining NAN rows, which will be due to missing DMOs
count_before = len(transition_data)
transition_data = transition_data.dropna()
print(f'{count_before - len(transition_data)} rows have missing DMOs and will be dropped...')

num_progressions = len(transition_data[transition_data['progression'] == 1])
unique_sub_progressions = len(transition_data[transition_data['progression'] == 1][participant_id_var].unique())
total_subs = transition_data[participant_id_var].nunique()

print(
    f'We have {num_progressions} progressions across {unique_sub_progressions} subjects, out of a total of {len(transition_data)} differences between timepoints across {total_subs} subjects.')

print(transition_data[transition_data['progression'] == 1].head(5))

show_transition_matrix(transition_data, 'trimmed_transitions')

transition_data = transition_data.drop(['next_edss', 'Next_Timepoint'], axis=1)

# drop processing columns

#####################################################
# Prepare Training Data
#
#####################################################

if data_row_type == 'transition':
    data = transition_data
elif data_row_type == 'participant':
    data = participant_data
elif data_row_type == 'progression':
    data = progression_data
else:
    raise ValueError('Unknown data_row_type:', data_row_type)

# randomise the rows, to ensure there's no data ordering bias
index_names = data.index.names
if index_names[0] is None:
    index_names = ['index']
data = data.sample(frac=1, random_state=random_state).reset_index()
data.set_index(index_names, inplace=True)

# drop irrelevant or 'cheating' columns
y_data = data['progression']  #.reset_index(drop=True)  # can be data['progression'] or data['multiclass_progression']

labels_X_all = data

non_dmo_features = ['edfscr1l', 'age', 'gender', 'time_since_diag', 'time_since_onset']
if drop_DMO_features:
    X_data = data[non_dmo_features]  #.reset_index(drop=True)
else:
    features_to_drop = [participant_id_var, 'visit.number', 'next_edss', 'progression']
    if drop_non_DMO_features:
        features_to_drop += non_dmo_features
    X_data = data.drop(features_to_drop, axis=1)

if drop_features_as_predictors:
    X_data = X_data.drop(drop_features_as_predictors, axis=1)

print(X_data.head(5))


#####################################################
# Data normalisation
#
#####################################################

# Data normalisation (Note the impact of this needs to be tested)
# e.g. StandardScaler, MinMaxScaler, RobustScaler
# If columns have different (e.g. skewed) distributions that may need different normalisations
# e.g. Log normalisation
# Note that StandardScaler is generally preferred,
# however combined with small data will hang SVC, while MinMaxScaler works

# moved scaler to inside function to prevent re-fitting

if show_data_distributions:
    for i, column in enumerate(X_data.columns):
        if X_data[column].dtype == 'float64':
            sns.displot(data=X_data, x=column, kind='kde')
        else:
            sns.displot(X_data, x=column)


######################################################################
# Initial parameter for Machine Learning models
# Initially these are the default values for the models, which are set to be generically applicable and avoid overfitting.
# Once the optimum parameter values for this dataset have been determined these can replace the default values
######################################################################

def get_fixed_features():
    fixed_feature_sets = {}
    csv_files = list(config_path.glob("*feature_rankings_G[0-3].csv"))
    for file in csv_files:
        group = str(file).split(".")[-2].split("_")[-1]
        fixed_feature_sets.update({group: {}})
        best_features_csv = pd.read_csv(os.path.join(config_path, "all_model_feature_rankings_" + group + ".csv"))
        relevant_cols = best_features_csv.columns.drop(["Rank", "Feature"])
        for col in relevant_cols:
            name = col.split("_")[-1]
            features = best_features_csv[best_features_csv[col].isna() == False]["Feature"].to_list()
            fixed_feature_sets[group].update({
                name: features,
            })

    return fixed_feature_sets


def get_initial_params():
    initial_params = {}
    hp_dtypes_dict = {
        "classifier__n_neighbors": 'Int64',
        "classifier__p": 'Int64',
        "classifier__leaf_size": 'Int64',
        "classifier__max_depth": 'Int64',
        "classifier__min_samples_split": 'Int64',
        "classifier__min_samples_leaf": 'Int64',
        "classifier__n_estimators": 'Int64',
        "classifier__min_child_weight": 'Int64'
    }
    converters = {
        'classifier__hidden_layer_sizes': lambda x: np.nan if x == '' else [int(float(i)) for i in x.translate(
            {ord(c): None for c in '()'}).split(',') if i]
    }
    csv_files = list(config_path.glob("*results_log_G[0-3].csv"))
    for file in csv_files:
        group = str(file).split(".")[-2].split("_")[-1]
        initial_params.update({group: {}})
        file_path = os.path.join(config_path, "hpt_results_log_" + group + ".csv")
        best_params_csv = pd.read_csv(file_path,
                                      dtype=hp_dtypes_dict,
                                      converters=converters)
        relevant_cols = best_params_csv.drop(columns=["search_type", "best_score"])
        relevant_cols.columns = [col.split("__")[-1] for col in relevant_cols.columns]
        for index, row in relevant_cols.iterrows():
            row = row.dropna()
            name = row["model_type"]
            features = row.drop(["model_type"])
            initial_params[group].update({name: {}})
            for feature, value in features.items():
                initial_params[group][name].update({
                    feature: value
                })

    return initial_params


# If set to empty [] the default parameter settings are used
fixed_feature_sets = get_fixed_features() if feature_selection_type == 'fixed' else []
print(fixed_feature_sets)
print()
initial_params = get_initial_params() if not do_hpt else []
print(initial_params)



######################################################################
# ML Model creation and Hyper Parameter Tuning (HPT)
#
# For each ML model the search space is defined. This can be categorical, integer or real values. These can be specified as sets of values or min/max ranges (with various distributions) for integers and reals.
# For Grid Search set of values are required
# For approximate search methods ranges can be used to search
# Real for skopt does not limit the decimal places or significant figures which leads to a potential infinite number of values.
# TODO find a way to limit reals by decimal places (see SKDecimal) and/or significant figures
# TODO test using PowerTransformer (e.g. yeo-johnson) before scaling, which may be useful for some models
######################################################################

default_params = initial_params['_'] if '_' in initial_params else []
group_params = initial_params[edss_group_type] if edss_group_type in initial_params else []


def set_params(model, params):
    for k, v in params.items():
        model.set_params(**{k: v})


def setInitialParams(model, type):
    if type in default_params and default_params[type] is not None:
        set_params(model, default_params[type])

    if type in group_params and group_params[type] is not None:
        set_params(model, group_params[type])


def add_prefix_to_keys(prefix, dicts):
    ret_dicts = []
    for d in dicts:
        ret_dicts.append({f'{prefix}{k}': v for k, v in d.items()})


svm_Cs = [0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
svm_gammas = ['scale', 'auto', 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
svm_degrees = [1, 2, 3, 4, 5]

# weight classes evenly, or use RandomOverSampler if parameter not available. Use 'None' to disable class weighting.
class_weight = 'balanced'


def get_svm():
    svc = SVC(random_state=random_state, class_weight=class_weight)
    setInitialParams(svc, 'SVM')

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', svc)
    ])

    return pipeline


# Note possibly remove 'poly' as an option from the SVC kernel to make it work on small data sets.
# See https://github.com/scikit-learn/scikit-learn/issues/4648
def get_svm_search_params(prefix=None):
    if prefix is None:
        prefix = 'classifier__'
    return [
        {prefix + 'C': svm_Cs, prefix + 'kernel': ['linear']},
        {prefix + 'C': svm_Cs, prefix + 'kernel': ['rbf'], prefix + 'gamma': ['scale']},
        {prefix + 'C': svm_Cs, prefix + 'kernel': ['sigmoid'], prefix + 'gamma': svm_gammas},
        {prefix + 'C': svm_Cs, prefix + 'kernel': ['poly'], prefix + 'gamma': svm_gammas,
         prefix + 'degree': svm_degrees}
    ]


def get_knn():
    # KNeighborsClassifier has neither class_weight nor sample_weight therefore use over-sampling
    knn = KNeighborsClassifier()
    setInitialParams(knn, 'KNN')

    return Pipeline([
        ('sampler', RandomOverSampler(random_state=random_state)),
        ('scaler', MinMaxScaler()),
        ('classifier', knn)
    ]) if class_weight == 'balanced' else Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', knn)
    ])


knn_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
knn_ps = [1, 2]
knn_leaf_sizes = [5, 10, 20, 30, 40, 50]
knn_metrics = [
    'euclidean']  #, 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'sqeuclidean']
knn_kdtree_metrics = ['euclidean']  #, 'cityblock', 'chebyshev']
knn_balltree_metrics = ['euclidean']  #, 'cityblock', 'chebyshev', 'canberra', 'braycurtis']


def get_knn_search_params(prefix=None):
    if prefix is None:
        prefix = 'classifier__'
    return [
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['uniform'],
         prefix + 'algorithm': ['brute']},
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['uniform'],
         prefix + 'algorithm': ['ball_tree', 'kd_tree'], prefix + 'leaf_size': knn_leaf_sizes},
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['distance'],
         prefix + 'metric': ['minkowski'], prefix + 'p': knn_ps,
         prefix + 'algorithm': ['brute']},
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['distance'],
         prefix + 'metric': ['minkowski'], prefix + 'p': knn_ps,
         prefix + 'algorithm': ['ball_tree', 'kd_tree'], prefix + 'leaf_size': knn_leaf_sizes},
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['distance'],
         prefix + 'metric': knn_metrics, prefix + 'algorithm': ['brute']},
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['distance'],
         prefix + 'metric': ['minkowski'], prefix + 'p': knn_ps,
         prefix + 'algorithm': ['ball_tree'], prefix + 'leaf_size': knn_leaf_sizes},
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['distance'],
         prefix + 'metric': knn_balltree_metrics, prefix + 'algorithm': ['ball_tree'],
         prefix + 'leaf_size': knn_leaf_sizes},
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['distance'],
         prefix + 'metric': ['minkowski'], prefix + 'p': knn_ps,
         prefix + 'algorithm': ['kd_tree'], prefix + 'leaf_size': knn_leaf_sizes},
        {prefix + 'n_neighbors': knn_neighbors, prefix + 'weights': ['distance'],
         prefix + 'metric': knn_kdtree_metrics, prefix + 'algorithm': ['kd_tree'],
         prefix + 'leaf_size': knn_leaf_sizes}
    ]


# MLPClassifier does not have class_weight and runs very slow if over-sampling is used
# Use this override hack to apply sample_weights to the fit() method instead
class MLPClassifierWrapper(MLPClassifier):

    # only override the fit() function here
    def _fit(self, X, y, sample_weight=None, incremental=False):
        if sample_weight is None and class_weight == 'balanced':
            sample_weight = compute_sample_weight(class_weight='balanced', y=y)
        super()._fit(X, y, sample_weight=sample_weight, incremental=incremental)


def get_mlp():
    mlp = MLPClassifierWrapper(random_state=random_state, max_iter=1000)
    setInitialParams(mlp, 'MLP')

    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', mlp)
    ])


def get_mlp_search_params(prefix=None):
    if prefix is None:
        prefix = 'classifier__'
    return [
        {prefix + 'hidden_layer_sizes': [50, 100],
         prefix + 'activation': ['logistic', 'tanh', 'relu'],
         prefix + 'solver': ['lbfgs', 'sgd', 'adam'],
         prefix + 'alpha': [0.0001, 0.001, 0.01, 0.05],
         prefix + 'learning_rate': ['constant', 'invscaling', 'adaptive'],
         prefix + 'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]  # Initial learning rate
         }
    ]


def get_lr():
    lr = LogisticRegression(random_state=random_state, solver='saga', max_iter=5000, class_weight=class_weight)
    setInitialParams(lr, 'LR')

    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', lr)
    ])


def get_lr_search_params(prefix=None):
    if prefix is None:
        prefix = 'classifier__'
    return [
        {prefix + 'solver': ['liblinear'],
         prefix + 'penalty': ['l1', 'l2'],
         prefix + 'C': [0.1, 0.5, 1.0, 1.5, 2.0, 3.0],  # Inverse of regularization strength
         prefix + 'fit_intercept': [True, False]
         },
        {prefix + 'solver': ['sag', 'lbfgs', 'newton-cg'],
         prefix + 'penalty': ['l2'],
         prefix + 'C': [0.1, 0.5, 1.0, 1.5, 2.0, 3.0],  # Inverse of regularization strength
         prefix + 'fit_intercept': [True, False]
         },
        {prefix + 'solver': ['saga'],
         prefix + 'penalty': ['elasticnet'],
         prefix + 'C': [0.1, 0.5, 1.0, 1.5, 2.0, 3.0],  # Inverse of regularization strength
         prefix + 'fit_intercept': [True, False],
         prefix + 'l1_ratio': [0, 0.5, 1]
         }
    ]


def get_dt():
    dt = DecisionTreeClassifier(random_state=random_state, class_weight=class_weight)
    setInitialParams(dt, 'DT')

    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', dt)
    ])


def get_dt_search_params(prefix=None):
    if prefix is None:
        prefix = 'classifier__'
    return [
        {prefix + 'criterion': ['gini', 'entropy'],
         prefix + 'max_depth': [5, 10, 15],
         prefix + 'min_samples_split': [2, 4, 6],
         prefix + 'min_samples_leaf': [1, 3, 5],
         prefix + 'max_features': ['sqrt', 'log2', None],
         prefix + 'ccp_alpha': [0.0, 0.1]  # Cost-Complexity Pruning alpha
         }
    ]


def get_rf():
    rf = RandomForestClassifier(random_state=random_state, class_weight=class_weight)
    setInitialParams(rf, 'RF')

    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', rf)
    ])


def get_rf_search_params(prefix=None):
    if prefix is None:
        prefix = 'classifier__'
    return [
        {prefix + 'n_estimators': [50, 100, 200],
         prefix + 'max_depth': [5, 10, 15],
         prefix + 'min_samples_split': [2, 4, 6],
         prefix + 'min_samples_leaf': [1, 3, 5],
         prefix + 'max_features': ['sqrt', 'log2', None],  # Number of features to consider at each split
         prefix + 'bootstrap': [True, False],
         prefix + 'criterion': ['gini', 'entropy']
         }
    ]


def get_xgb():
    xgb = XGBClassifier(random_state=random_state, use_label_encoder=False,
                        eval_metric='logloss', class_weight=class_weight)
    setInitialParams(xgb, 'XGB')

    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', xgb)
    ])


def get_xgb_search_params(prefix=None):
    if prefix is None:
        prefix = 'classifier__'
    return [
        {prefix + 'n_estimators': [50, 100, 200, 500, 1000],
         prefix + 'learning_rate': [0.6, 0.8, 1.0],
         prefix + 'max_depth': [3, 5, 7, 9],
         prefix + 'min_child_weight': [3, 5, 7, 9],
         prefix + 'gamma': [i / 10.0 for i in range(0, 5)],  # Min loss reduction for split
         prefix + 'subsample': [i / 10.0 for i in range(6, 10)],  # Fraction of samples used for fitting
         prefix + 'colsample_bytree': [i / 10.0 for i in range(6, 10)],  # Fraction of features used per tree
         prefix + 'reg_alpha': [1e-5, 1e-2, 0.1],  # L1 regularization
         prefix + 'reg_lambda': [1e-5, 1e-2, 0.1]  # L2 regularization
         }
    ]


def get_model(model_type):
    match model_type:
        case 'SVM':
            return get_svm()
        case 'XGB':
            return get_xgb()
        case 'KNN':
            return get_knn()
        case 'MLP':
            return get_mlp()
        case 'DT':
            return get_dt()
        case 'LR':
            return get_lr()
        case 'RF':
            return get_rf()
        case 'XGB':
            return get_xgb()
        case _:
            raise Exception('Unknown model type: ', model_type)


def get_model_search_params(model_type, prefix=None):
    match model_type:
        case 'SVM':
            return get_svm_search_params(prefix=prefix)
        case 'KNN':
            return get_knn_search_params(prefix=prefix)
        case 'MLP':
            return get_mlp_search_params(prefix=prefix)
        case 'DT':
            return get_dt_search_params(prefix=prefix)
        case 'LR':
            return get_lr_search_params(prefix=prefix)
        case 'RF':
            return get_rf_search_params(prefix=prefix)
        case 'XGB':
            return get_xgb_search_params(prefix=prefix)
        case _:
            raise Exception('Unknown model type: ', model_type)



######################################################################
# Hyper Parameter Tuning
#
# Alternative to BayesSearchCV:
#   catCMA by Optuna https://hub.optuna.org/samplers/catcma/
#.  Hyperopt
######################################################################

def get_sfs_bayes_search(model, params):
    return BayesSearchCV(
        model,
        params,
        n_iter=sfs_hpt_n_iter,
        random_state=random_state,
        verbose=sfs_hpt_verbose,
        n_jobs=sfs_hpt_n_jobs,
        cv=RepeatedStratifiedKFold(n_splits=n_splits,
                                   n_repeats=n_repeats,
                                   random_state=random_state),
        scoring=scoring
    )


def get_bayes_search(model, params, n_iter=hpt_n_iter):
    return BayesSearchCV(
        model,
        params,
        n_iter=n_iter,
        random_state=random_state,
        verbose=hpt_verbose,
        n_jobs=hpt_n_jobs,
        cv=RepeatedStratifiedKFold(n_splits=n_splits,
                                   n_repeats=n_repeats,
                                   random_state=random_state),
        scoring=scoring
    )


def get_grid_search(model, params):
    return GridSearchCV(
        model,
        params,
        verbose=hpt_verbose,
        n_jobs=hpt_n_jobs,
        cv=RepeatedStratifiedKFold(n_splits=n_splits,
                                   n_repeats=n_repeats,
                                   random_state=random_state),
        scoring=scoring
    )

######################################################################
# Machine learning models
#
######################################################################

models = {}

for model_type in model_types:
    models[model_type] = get_model(model_type)
    if sfs_do_hpt:
        models[model_type] = get_sfs_bayes_search(models[model_type], get_model_search_params(model_type))


###################################################################
# Utility functions for cross-validation and results
###################################################################

def has_parameter(obj, param_name):
    sig = inspect.signature(obj)
    for param in sig.parameters.values():
        if param.name is param_name:
            return True
    return False


def get_score(model, X_data, y_data, n_splits, n_repeats):
    cm, y_test, y_pred, y_pred_prob, fold_metrics, iteration_metrics = do_cv(
        model, X_data, y_data, n_splits=n_splits, n_repeats=n_repeats, quiet=True)
    return np.mean(iteration_metrics[scoring])


def replace_index(things, index):
    if isinstance(things, pd.Series):
        things.index = index
    else:
        things = pd.Series(things, index)
    return things


def get_metric_dict() -> dict:
    return {
        'sensitivity': [],
        'specificity': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'auc': []
    }


def do_cv(model, X_data, y_data, n_splits=n_splits, n_repeats=n_repeats,
          random_state=random_state, quiet=False, probability=False):
    # requires: set_config(transform_output='pandas')
    # transformer = ColumnTransformer([('standard_scaler', ss, feature_cols)],
    #                                 remainder='passthrough',
    #                                 verbose_feature_names_out=False)

    iteration_metrics = get_metric_dict()
    fold_metrics = get_metric_dict()
    combined_cm = None
    combined_y_test = None
    combined_y_pred = None
    combined_y_pred_prob = None
    all_folds_y_test = None
    all_folds_y_pred = None
    all_folds_y_pred_prob = None
    kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for i, (train_split_indices, test_split_indices) in enumerate(kf.split(X_data, y_data)):
        fold = i % n_splits
        iteration = int(i / n_splits)

        X_train_split = X_data.iloc[train_split_indices]
        X_test_split = X_data.iloc[test_split_indices]
        y_train_split = y_data.iloc[train_split_indices]
        y_test_split = y_data.iloc[test_split_indices]

        model.fit(X_train_split, y_train_split)

        y_pred = model.predict(X_test_split)
        y_pred_prob = model.predict_proba(X_test_split)[:, 1] if probability else None

        # Add fold specific index
        # check if data already has fold index
        if isinstance(y_test_split.index, pd.MultiIndex):
            y_test_index = y_test_split.index
        else:
            tuples = [(id, fold, iteration) for id in y_test_split.index]
            y_test_index = pd.MultiIndex.from_tuples(tuples, names=["id", "fold", "iteration"])
            y_test_split.index = y_test_index
        # turn y_pred into a series
        y_pred = replace_index(y_pred, y_test_index)
        if not y_pred_prob is None:
            # turn y_pred_prob into a series
            y_pred_prob = replace_index(y_pred_prob, y_test_index)

        metrics = get_metrics_from_prediction(y_test_split, y_pred, y_pred_prob)
        for k in metrics.keys():
            fold_metrics[k].append(metrics[k])

        if not quiet:
            print(f'Fold {i}')
            print(f'accuracy: {metrics["accuracy"]:.2f}')
            print(f'precision_macro: {metrics["precision_macro"]:.2f}')
            print(f'recall_macro: {metrics["recall_macro"]:.2f}')
            print(f'f1_macro: {metrics["f1_macro"]:.2f}')

        cm = confusion_matrix(y_test_split, y_pred)
        combined_cm = cm if combined_cm is None else combined_cm + cm
        combined_y_test = y_test_split if combined_y_test is None else pd.concat([combined_y_test, y_test_split])
        combined_y_pred = y_pred if combined_y_pred is None else pd.concat([combined_y_pred, y_pred])
        if y_pred_prob is not None:
            combined_y_pred_prob = y_pred_prob if combined_y_pred_prob is None \
                else pd.concat([combined_y_pred_prob, y_pred_prob])

        # concatenate fold predictions for this iteration
        all_folds_y_test = y_test_split if all_folds_y_test is None else pd.concat([all_folds_y_test, y_test_split])
        all_folds_y_pred = y_pred if all_folds_y_pred is None else pd.concat([all_folds_y_pred, y_pred])
        if y_pred_prob is not None:
            all_folds_y_pred_prob = y_pred_prob if all_folds_y_pred_prob is None \
                else pd.concat([all_folds_y_pred_prob, y_pred_prob])
        # check if we have done all the folds in the current iteration
        if (i != 0 and (i + 1) % n_splits == 0):
            # calculate metrics for this iteration
            metrics = get_metrics_from_prediction(all_folds_y_test, all_folds_y_pred, all_folds_y_pred_prob)
            for k in metrics.keys():
                iteration_metrics[k].append(metrics[k])
            all_folds_y_test = None
            all_folds_y_pred = None
            all_folds_y_pred_prob = None

    if not quiet:
        print('-' * 50)
        print(
            f'mean accuracy = {np.mean(fold_metrics["accuracy"]):.2f}' + u"\u00B1" + f'{np.std(fold_metrics["accuracy"]):.2f}')
        print(
            f'mean macro precision = {np.mean(fold_metrics["precision_macro"]):.2f}' + u"\u00B1" + f'{np.std(fold_metrics["precision_macro"]):.2f}')
        print(
            f'mean macro recall = {np.mean(fold_metrics["recall_macro"]):.2f}' + u"\u00B1" + f'{np.std(fold_metrics["recall_macro"]):.2f}')
        print(
            f'mean macro f1 = {np.mean(fold_metrics["f1_macro"]):.2f}' + u"\u00B1" + f'{np.std(fold_metrics["f1_macro"]):.2f}')
        print('-' * 50)
    return combined_cm, combined_y_test, combined_y_pred, combined_y_pred_prob, fold_metrics, iteration_metrics


def display_cm(cm, fig_name=None, class_names=None):
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_perc = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    labels = np.array([
        f"{count}\n({perc:.1%})" for count, perc in zip(cm.flatten(), cm_perc.flatten())
    ]).reshape(cm.shape)

    df_cm = pd.DataFrame(cm_perc, index=class_names, columns=class_names)

    plt.figure(figsize=(7, 7))
    ax = plt.subplot()

    sns.heatmap(df_cm, annot=labels, fmt='', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 14})

    ax.set_xlabel('Predicted labels', fontsize=14)
    ax.set_ylabel('True labels', fontsize=14)

    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    ax.tick_params(axis='both', which='major', labelsize=12)

    if fig_name:
        plt.title(fig_name, fontsize=16)
        plt.savefig(f"{fig_save_dir}{fig_name}_{edss_group_type}.png", bbox_inches='tight')

    plt.show()


def show_evaluation(model, X_test, y_test, fig_name=None):
    y_pred = model.predict(X_test)

    print(f'accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    display_cm(cm, fig_name=fig_name)


def get_metrics_from_cm(cm, type='micro', dp=3):
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if type == 'micro':
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        precision = (pos_precision + neg_precision) / 2.0
        pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall = (pos_recall + neg_recall) / 2.0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'type': type,
        'accuracy': round(accuracy, dp),
        'precision': round(precision, dp),
        'recall': round(recall, dp),
        'f1': round(f1_score, dp)
    }


def get_metrics_from_prediction(actual, predicted, score):
    return {
        'sensitivity': sensitivity_score(actual, predicted),  # default to pos_label=1
        'specificity': specificity_score(actual, predicted),  # default to pos_label=1
        'accuracy': accuracy_score(actual, predicted),
        'precision': precision_score(actual, predicted),
        'recall': recall_score(actual, predicted),
        'f1': f1_score(actual, predicted),
        'precision_macro': precision_score(actual, predicted, average='macro', zero_division=0),
        'recall_macro': recall_score(actual, predicted, average='macro', zero_division=0),
        'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
        'auc': roc_auc_score(actual, score, average='macro') if score is not None else 0
    }

###################################################################
# Functions for feature selection and analysis
#
###################################################################

def rf_feature_analysis(X, y, random_state=random_state):
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X, y)

    # The importance of a feature is how much this feature is used in each tree of the forest.
    # It is computed as the (normalized) total reduction of the criterion brought by that feature.
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    fig, ax = plt.subplots()
    importances = pd.Series(model.feature_importances_, index=np.array(X.columns))
    importances = pd.Series(data=importances, index=X.columns).sort_values(ascending=False)
    importances.plot.bar(yerr=std, ax=ax)
    ax.set_title('Feature importance using MDI')
    ax.set_ylabel('Mean decrease in impurity')
    fig.tight_layout()

    return importances.index.tolist()


def rf_perm_feature_analysis(X, y, random_state=random_state):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X, y)

    result = permutation_importance(model, X, y, n_repeats=10, random_state=random_state, n_jobs=fs_n_jobs)

    fig, ax = plt.subplots()
    importances = pd.Series(result.importances_mean, index=np.array(X.columns))
    importances = pd.Series(data=importances, index=X.columns).sort_values(ascending=False)
    importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title('Feature importance using permutation on full model')
    ax.set_ylabel('Mean accuracy decrease')
    fig.tight_layout()
    plt.savefig(fig_save_dir + 'rm_perm_feature_analysis' + '_' + edss_group_type + '.png')
    plt.show()

    return importances.index.tolist()


def plot_permutation_importance(clf, X, y, ax, indexes=None, random_state=random_state):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=random_state, n_jobs=fs_n_jobs)
    if indexes is not None:
        perm_sorted_idx = indexes
    else:
        perm_sorted_idx = result.importances_mean.argsort()

    tick_labels_dict = {'tick_labels': X.columns[perm_sorted_idx]}
    ax.boxplot(result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
    ax.axvline(x=0, color='k', linestyle='--')
    return ax


def rf_combine_feature_analysis(X, y, random_state=random_state):
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X, y)

    mdi_importances = pd.Series(clf.feature_importances_, index=X.columns)
    tree_importance_sorted_idx = np.argsort(clf.feature_importances_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    mdi_importances.sort_values().plot.barh(ax=ax1)
    ax1.set_xlabel('Gini importance')
    plot_permutation_importance(clf, X, y, ax2, tree_importance_sorted_idx)
    ax2.set_xlabel('Decrease in accuracy score')
    fig.suptitle(
        'Impurity-based vs. permutation importance on multicollinear features (train set)'
    )
    _ = fig.tight_layout()
    plt.savefig(fig_save_dir + 'rm_combined_feature_analysis' + '_' + edss_group_type + '.png')
    plt.show()

    return mdi_importances.sort_values(ascending=False).index.tolist()


def mi_feature_selection(X, y, mi_threshold=0.0, random_state=random_state):
    # feature importance analysis
    pearson_corr = X.corrwith(y).sort_values(ascending=False)
    print(pearson_corr)

    # mutual information gain for feature removal
    mi = mutual_info_classif(X, y, random_state=random_state, n_jobs=fs_n_jobs)
    mi = pd.Series(data=mi, index=X.columns).sort_values(ascending=False)
    mi.plot(kind='bar')
    plt.title('Feature importance via mutual information')
    plt.savefig(fig_save_dir + 'mi_feature_selection' + '_' + edss_group_type + '.png')
    plt.show()

    ranking = mi.index.tolist()

    # delete features with 0 information gain
    mi = mi[lambda x: x > mi_threshold]
    X_filtered = X[mi.index]

    return X_filtered, ranking


# Feature importance from coefficients
def fi(X, y, model, threshold=0.0):
    model.fit(X, y)
    feature_names = np.array(X.columns)
    importance = pd.Series(data=np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_),
                           index=feature_names).sort_values(ascending=False)
    importance.plot(kind='bar')
    plt.title('Feature importance via coefficients: ' + type(model).__name__)
    plt.savefig(fig_save_dir + 'feature_importance_coeff' + '_' + edss_group_type + '.png')
    plt.show()

    sfm = SelectFromModel(model, threshold=threshold).fit(X, y)
    print(f'Features selected by SelectFromModel above threshold {threshold}: {feature_names[sfm.get_support()]}')

    # NEW: return ranking
    return importance.index.tolist()


def show_correlation(X):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).statistic

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    _ = fig.tight_layout()
    plt.savefig(fig_save_dir + 'spearman_correlation' + '_' + edss_group_type + '.png')
    plt.show()

    return dist_linkage


def cluster_selection(X, y, dist_linkage, threshold=0.05, random_state=random_state):
    # Cluster selection groups features rather than strictly ranking all of them by importance.
    # Left intact without a global ranking return.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=0.5)

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    print(f'Baseline accuracy on test data: {clf.score(X_test, y_test):.2}')

    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = X.columns[selected_features]

    X_train_sel = X_train[selected_features_names]
    X_test_sel = X_test[selected_features_names]

    clf_sel = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf_sel.fit(X_train_sel, y_train)
    print(
        'Baseline accuracy on test data with features removed:'
        f' {clf_sel.score(X_test_sel, y_test):.2}'
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_permutation_importance(clf_sel, X_test_sel, y_test, ax)
    ax.set_title('Permutation Importance on selected subset of features\n(test set)')
    ax.set_xlabel('Decrease in accuracy score')
    ax.figure.tight_layout()
    plt.savefig(fig_save_dir + 'cluster_selection' + '_' + edss_group_type + '.png')
    plt.show()

    return X[selected_features_names]


def pca_selection(X, random_state=random_state):
    # normalisation
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95, svd_solver='full', random_state=random_state)  # keep 95% of variance
    X_pca = pd.DataFrame(data=pca.fit_transform(X_scaled))
    print('Explained variance:', pca.explained_variance_ratio_)
    print('Cumulative:', np.cumsum(pca.explained_variance_ratio_))
    print('Shape of Original Dataset:', X_scaled.shape)
    print('Shape after PCA:', X_pca.shape)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', yticklabels=np.array(X.columns))
    plt.title('Feature Importance in Principal Components')
    plt.savefig(fig_save_dir + 'pca' + '_' + edss_group_type + '.png')
    plt.show()

    # NEW: Derive ranking based on absolute loadings sum
    loadings_abs_sum = np.abs(pca.components_).sum(axis=0)
    ranking = X.columns[np.argsort(loadings_abs_sum)[::-1]].tolist()

    return X_pca, ranking


def get_rfecv(estimator, n_splits=n_splits, scoring=scoring, min_features_to_select=3, n_jobs=fs_n_jobs):
    return RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=n_jobs
    )


# Recursive Feature Elimination
# Find the optimal set of features using models (e.g. DT, RF, GBM) which use inherent feature selection
def rfe(X, y, model):
    model.fit(X, y)

    print(f'Optimal number of features: {model.n_features_}')

    # summarize all features
    feature_names = np.array(X.columns)
    for i in range(X.shape[1]):
        print('%s, Selected %s, Rank: %d' % (feature_names[i], model.support_[i], model.ranking_[i]))

    rfe_data = {
        key: value
        for key, value in model.cv_results_.items()
        if key in ['n_features', 'mean_test_score', 'std_test_score']
    }
    cv_results = pd.DataFrame(rfe_data)
    plt.figure()
    plt.xlabel('Number of features selected')
    plt.ylabel('Mean test ' + scoring)
    plt.errorbar(
        x=cv_results['n_features'],
        y=cv_results['mean_test_score'],
        yerr=cv_results['std_test_score'],
    )
    plt.title('Recursive Feature Elimination \n' + model.estimator_.__class__.__name__)
    plt.savefig(fig_save_dir + 'rfe' + '_' + edss_group_type + '.png')
    plt.show()

    # NEW: Create ranking from RFE ranking array
    ranking = pd.Series(model.ranking_, index=X.columns).sort_values().index.tolist()

    return model.transform(X), ranking


# Test the various Recursive Feature Elimination (RFE) approaches (DT, RF, GBM) to selecting features
# and applying those features to the specified model
def rfe_test(X, y, model):
    # get a list of models to evaluate
    rfe_models = dict()
    rfe_models['dt'] = get_rfecv(DecisionTreeClassifier(random_state=random_state))
    rfe_models['rf'] = get_rfecv(RandomForestClassifier(random_state=random_state))
    rfe_models['gbm'] = get_rfecv(GradientBoostingClassifier(random_state=random_state))

    for name, rfe_model in rfe_models.items():
        # Unpack the new tuple return from rfe
        rfe_X, ranking = rfe(X, y, rfe_model)
        cm, actual, predicted, score, fold_metrics, iteration_metrics = do_cv(model, rfe_X, y, quiet=True)
        print('>%s %s' % (name, rfe_X.shape))
        for metric in metrics.keys():
            print(f'{metric}: {np.mean(iteration_metrics[metric])}')


def get_model_name(model):
    model_name = model.__class__.__name__

    try:
        if model_name == 'Pipeline':
            classifier = model['classifier']
            if classifier:
                model_name = classifier.__class__.__name__
        if hasattr(model, 'estimator'):
            estimator = model.estimator
            if estimator:
                estimator_name = estimator.__class__.__name__
                model_name += ': ' + estimator_name
                if estimator_name == 'Pipeline':
                    classifier = estimator['classifier']
                    if classifier:
                        model_name += ': ' + classifier.__class__.__name__
    except Exception as e:
        # There's an issue with RandomForestClassifier which throws an annoying exception
        warning = RuntimeWarning(*e.args)
        warning.with_traceback(e.__traceback__)
        warnings.warn(warning)

    return model_name


SUFFIXES = {1: 'st', 2: 'nd', 3: 'rd'}


def get_ordinal_suffix(i):
    if 10 <= i % 100 <= 20:
        return 'th'
    else:
        return SUFFIXES.get(i % 10, 'th')


# This function extends SFS in an attempt to prevent the search getting stuck in local maxima.
# This approach performs multiple restarts
# TODO extend SFS so that rather than selecting the best feature at each step,
# TODO stochastically select a feature from the top k best features
def sfs2(X, y, model, scoring=scoring, n_splits=n_splits, n_repeats=n_repeats,
         n_jobs=fs_n_jobs, random_state=random_state, forward=sfs_forward, floating=sfs_floating):
    results = []
    ranks = {}
    for i in range(0, sfs2_restarts):
        score, X_data, feature_rank = sfs(X, y, model, scoring=scoring, n_splits=n_splits, n_repeats=sfs2_repeats,
                                          n_jobs=n_jobs, verbose=sfs2_verbose, random_state=random_state + i,
                                          forward=forward, floating=floating, min_k=1, max_k=sfs2_max_k,
                                          show_plot=False)
        print('%d/%d: best (%s: %.3f): %s' % (i + 1, sfs2_restarts, scoring, score, list(X_data.columns.values)))

        results.append(X_data.columns.values)
        ranks[i] = feature_rank

    # remove duplicate results
    # result.sort() caused error: 'StringArray' object has no attribute 'sort'
    # This is an open bug in Panda: https://github.com/pandas-dev/pandas/issues/64977
    unique_results = []
    for result in results:
        sorted_result = sorted(result)
        if not any(np.array_equal(sorted_result, unique_result) for unique_result in unique_results):
            unique_results.append(sorted_result)

    # find best result on data with full number of repeated CV splits
    # use backward SFS search to reduce number of features if it improves results
    best_feature_set = []
    best_score = 0
    for idx, result in enumerate(unique_results):
        score, X_data, feature_rank = sfs(X[result], y, model, scoring=scoring, n_splits=n_splits, n_repeats=n_repeats,
                                          n_jobs=n_jobs, verbose=sfs2_verbose, random_state=random_state,
                                          forward=False, floating=floating, min_k=1, max_k=-1, show_plot=False)
        print(f'{idx + 1}/{len(unique_results)}: {score}: {list(X_data.columns.values)}')
        if score > best_score or (score == best_score and len(X_data.columns.values) < len(best_feature_set)):
            best_score = score
            best_feature_set = X_data.columns.values

    print(f'Best result = {best_score}: {list(best_feature_set)}')

    df_sfs_ranks = pd.DataFrame(ranks)
    # allocate rank to values which have not been selected
    df_sfs_ranks.fillna(len(X.columns), inplace=True)
    df_sfs_ranks['Average_SFS_Rank'] = df_sfs_ranks.mean(axis=1)
    df_sfs_ranks = df_sfs_ranks.sort_values(by='Average_SFS_Rank', ascending=True)
    print(df_sfs_ranks)
    feature_rank = {}
    i = 0
    for index, row in df_sfs_ranks.iterrows():
        i = i + 1
        feature_rank[index] = i

    return best_score, X[best_feature_set], feature_rank


def sfs(X, y, model, scoring=scoring, n_splits=n_splits, n_repeats=n_repeats,
        n_jobs=fs_n_jobs, verbose=sfs_verbose, random_state=random_state,
        forward=sfs_forward, floating=sfs_floating, min_k=sfs_min_features, max_k=sfs_max_features,
        show_plot=True):
    # Set the min/max features
    # min features will only affect the size of the selected best feature set, compute will still set at 1 feature
    # max will terminate the compute at max features, this may produce different results for the same max if floating is False/True
    if min_k < 1:
        min_k = 1
    if max_k < 1 or max_k > len(X.columns):
        max_k = len(X.columns)

    sfs = SFS(model,
              k_features=(min_k, max_k),
              forward=forward,
              floating=floating,
              verbose=verbose,
              scoring=scoring,
              n_jobs=n_jobs,
              cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state))

    sfs.fit(X, y)

    best_feature_set = [X.columns[i] for i in sfs.k_feature_idx_]

    if show_plot:
        # print('all subsets:\n', sfs.subsets_)
        plot_sfs(sfs.get_metric_dict(), kind='std_err')
        plt.title(
            f'SFS ({"forward" if forward else "backward"}, {"floating" if floating else ""}) for {get_model_name(model)} on {scoring}')
        plt.show()

    # calculate SFS feature rank using order they're added or removed
    # might be worth having a second pair of eyes have a look over the logic here
    feature_rank = {}
    pre_idx = set()

    # use a counter to ensure ranks are 1, 2, 3, etc.
    rank_counter = 1

    for i in sorted(sfs.subsets_.keys()):
        current_idx = set(sfs.subsets_[i]['feature_idx'])
        new_features = current_idx - pre_idx

        for feat_idx in new_features:
            feature_name = X.columns[feat_idx]

            if feature_name in best_feature_set and feature_name not in feature_rank:
                feature_rank[feature_name] = rank_counter if forward else (len(X.columns) + 1) - rank_counter
                rank_counter += 1

        pre_idx = current_idx

    return sfs.k_score_, X[best_feature_set], feature_rank


def auto_cluster_features_kmeans(X, max_clusters=10, feature_names=None, random_state=random_state):
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(np.shape(X)[1])]

    X_fit = X.T.values if isinstance(X, pd.DataFrame) else X.T

    n_features = X_fit.shape[0]
    max_k = min(max_clusters, n_features - 1)

    print()
    # Silhouette score requires at least 2 clusters
    if max_k < 2:
        print('Not enough features to cluster.')
        return {'Cluster 0': feature_names}

    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
        labels = kmeans.fit_predict(X_fit)

        # The Silhouette Score measures how similar an object is to its own cluster
        # compared to other clusters. Higher is better (ranges from -1 to 1).
        score = silhouette_score(X_fit, labels)

        if score > best_score:
            best_score = score
            best_k = k

    print(f'Optimal clusters found: {best_k} (Silhouette Score: {best_score:.3f})')

    final_kmeans = KMeans(n_clusters=best_k, n_init='auto', random_state=random_state)
    final_kmeans.fit(X_fit)

    feature_clusters = {f'Cluster {i}': [] for i in range(best_k)}
    for feature, label in zip(feature_names, final_kmeans.labels_):
        feature_clusters[f'Cluster {label}'].append(feature)

    return feature_clusters


def get_redundant_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()

    found = False
    print()
    print(f'--- Pairs with correlation > {threshold} ---')

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            score = corr_matrix.iloc[i, j]
            if score >= threshold:
                feat_a = corr_matrix.columns[i]
                feat_b = corr_matrix.columns[j]
                print(f'{feat_a} <--> {feat_b} : {score:.4f}')
                found = True

    if not found:
        print('No features found above this threshold.')



###################################################################
# Feature selection
# Note that in data preparation rows with NaN values are excluded.
# Removing these features (columns) may increase the data,
# if that feature is the reason for NaN exclusion.
###################################################################


collected_rankings = {}

if show_feature_agglomeration:
    auto_grouped_features = auto_cluster_features_kmeans(X_data, max_clusters=20)

    for cluster_name, features in auto_grouped_features.items():
        print(f'{cluster_name}: {features}')

if show_redundant_features:
    get_redundant_features(X_data)

if show_feature_analysis:
    # collected_rankings['RF_MDI'] = rf_feature_analysis(X_data, y_data)
    # collected_rankings['RF_Perm'] = rf_perm_feature_analysis(X_data, y_data)
    collected_rankings['RF_Combine'] = rf_combine_feature_analysis(X_data, y_data)

    ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5))
    collected_rankings['Ridge_Coeff'] = fi(X_data, y_data, ridge, threshold=0.0001)

if show_mi_feature_selection or feature_selection_type == 'mi':
    X_data_mi, collected_rankings['MI'] = mi_feature_selection(X_data, y_data, mi_threshold=mi_threshold)

if show_cluster_feature_selection or feature_selection_type == 'cluster':
    dist_linkage = show_correlation(X_data)
    X_data_cluster = cluster_selection(X_data, y_data, dist_linkage)

if show_pca_feature_selection or feature_selection_type == 'pca':
    # NEW: unpack the tuple
    X_data_pca, collected_rankings['PCA'] = pca_selection(X_data)

if show_rfe_feature_selection or feature_selection_type == 'rfe':
    # xgb = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    # X_data_rfe, collected_rankings['RFE_XGB'] = rfe(X_data, y_data, xgb)
    rfe_test(X_data, y_data, models['SVM'])

X_data_sfs = {}
if show_sfs_feature_selection or feature_selection_type == 'sfs' or feature_selection_type == 'sfs2':
    for model_type in model_types:
        if feature_selection_type == 'sfs':
            score, X_data_sfs[model_type], collected_rankings['SFS_' + model_type] = \
                sfs(X_data, y_data, models[model_type])
            print(f'Best result = {score}: {list(X_data_sfs[model_type].columns.values)}')

            # invert forward/backward and apply to best feature set
            new_forward = False if sfs_forward else True
            new_score, new_X_data, new_collected_rankings = \
                sfs(X_data_sfs[model_type], y_data, models[model_type], forward=new_forward)

            if new_score > score:
                print(f'New best result = {new_score}: {list(new_X_data.columns.values)}')
                # replace with improved results
                X_data_sfs[model_type] = new_X_data
                collected_rankings[
                    'SFS_' + model_type] = new_collected_rankings  # overwrite forward sfs rankings with backwards sfs rankings

        else:  #if feature_selection_type == 'sfs2'
            score, X_data_sfs[model_type], collected_rankings['SFS_' + model_type] = \
                sfs2(X_data, y_data, models[model_type])

    sfs_keys = [key for key in collected_rankings.keys() if key.startswith('SFS_')]

    sfs_ranks_dict = {}
    for key in sfs_keys:
        sfs_ranks_dict[key] = collected_rankings[key]

    df_sfs_ranks = pd.DataFrame(sfs_ranks_dict)
    df_sfs_ranks['Average_SFS_Rank'] = df_sfs_ranks.mean(axis=1)
    df_sfs_ranks = df_sfs_ranks.sort_values(by='Average_SFS_Rank', ascending=True)

model_fixed_features = {}
if feature_selection_type == 'fixed':
    if edss_group_type in fixed_feature_sets:
        model_fixed_features = fixed_feature_sets[edss_group_type]
elif feature_selection_type == 'cluster':
    X_data = X_data_cluster
elif feature_selection_type == 'mi':
    if mi_threshold >= 0:
        X_data = X_data_mi
    else:
        print(f'Feature selection type "mi", however mi_threshold is {mi_threshold}, so has no effect.')
elif feature_selection_type == 'pca':
    X_data = X_data_pca
elif feature_selection_type == 'sfs' or feature_selection_type == 'sfs2':
    X_data = X_data  # this gets changed later if selected so it doesn't really matter what we do here
# elif feature_selection_type == 'rfe':
#     X_data = X_data_rfe
elif feature_selection_type is not None:
    print('Feature selection type not recognized')


def get_x_data(model_type):
    match feature_selection_type:
        case 'fixed':
            if model_fixed_features[model_type] is not None:
                return X_data[model_fixed_features[model_type]]
        case 'sfs' | 'sfs2':
            return X_data_sfs[model_type]
    return X_data


# Dict to store everything about each model
# y_data might be redundant, but it makes life a bit easier later as now everything is in one dict
model_data_dict = {}
for model_type in model_types:
    model_data_dict[model_type] = {
        'model': models[model_type],
        'X_data': get_x_data(model_type),
        'y_data': y_data
    }

print('Feature selection:', feature_selection_type)
print('X_data shape:', X_data.shape)

if collected_rankings:
    df_rankings = pd.DataFrame(collected_rankings)
    df_rankings = df_rankings.rename_axis('Feature').reset_index()
    df_rankings.index = np.arange(1, len(df_rankings) + 1)
    df_rankings.index.name = 'Rank'
    print('\nCombined Feature Rankings:')
    print(df_rankings)
    df_rankings.to_csv(fig_save_dir + 'all_model_feature_rankings_' + edss_group_type + '.csv')


###################################################################
# Hyper Parameter Tuning of models
###################################################################

def do_search(model_type, replace_model=True):
    search_type = model_hpt_type[model_type]
    model = get_model(model_type)
    params = get_model_search_params(model_type)
    X_data = model_data_dict[model_type]['X_data']
    y_data = model_data_dict[model_type]['y_data']
    if search_type == 'grid':
        search = get_grid_search(model, params)
    elif search_type == 'bayes':
        search = get_bayes_search(model, params)
    else:
        raise Exception('Unknown search type: ', search_type)
    combos = list(ParameterGrid(params))
    print(f'\nPerform {search_type} search on {len(combos)} parameter combinations for {model_type}')
    if search_type == 'bayes':
        print(f'Using {hpt_n_iter} iterations')
    start = timer()
    search.fit(X_data, y_data)
    print(f'Finished {search_type} search in {timer() - start:.3f}s')
    current_model = model_data_dict[model_type]['model']
    current_params = {key: current_model.get_params()[key] for key in search.best_params_.keys()}

    cm, y_test, y_pred, y_pred_prob, fold_metrics, iteration_metrics = do_cv(current_model, X_data, y_data, quiet=True)
    score = np.mean(iteration_metrics[scoring])
    print(f'Current score: {score:.3} = {current_params}')
    print(f'HPT Best: {search.best_score_:.3} = {search.best_params_}')

    final_results = {
        'model_type': model_type,
        'search_type': search_type,
    }

    if replace_model and score < search.best_score_:
        final_results['best_score'] = round(search.best_score_, 4),
        model_data_dict[model_type]['model'] = search.best_estimator_
        final_results.update(search.best_params_)
    else:
        final_results['best_score'] = round(score, 4)
        final_results.update(current_params)

    final_results = pd.DataFrame([final_results])

    csv_filename = '../output/hpt_results_log_' + edss_group_type + '.csv'

    if not os.path.exists(csv_filename):
        final_results.to_csv(csv_filename, index=False)
        print(f"Created new log file and saved results to {csv_filename}")
    else:
        df_existing = pd.read_csv(csv_filename)
        df_combined = pd.concat([df_existing, final_results], ignore_index=True)
        df_combined.to_csv(csv_filename, index=False)
        print(f"Appended results to {csv_filename}")


if do_hpt:
    for model_type in model_types:
        if model_hpt_type[model_type] is not None:
            do_search(model_type)



######################################################################
# Final Classification / Prediction
#
######################################################################

model_performance = []
model_metrics = {}
model_prediction = {}

for model_name, params in model_data_dict.items():
    # do_cv does all this anyway
    # yeah I think this is redundant now.
    """
    params['X_train'], params['X_test'], params['y_train'], params['y_test'] = (
        train_test_split(params['X_data'], params['y_data'], stratify=params['y_data'], test_size=0.2, random_state=42))

    params['X_train'] = scaler.fit_transform(params['X_train'])
    params['X_test'] = scaler.transform(params['X_test'])

    # maybe redundant but helps with some formatting later
    params['sample_weights'] = compute_sample_weight(class_weight='balanced', y=params['y_train'])
    """

    # After this point we set probability=True for SVM to calculate AUC
    # NOTE: The probability parameter of sklearn.svm.SVC and sklearn.svm.NuSVC is deprecated due to not being thread-safe and will be removed in 1.11. Use sklearn.calibration.CalibratedClassifierCV with the respective estimator and ensemble=False instead.
    if model_name == 'SVM':
        set_params(params['model'], {'classifier__probability': True})
        print("params['model']", params['model'].get_params())

    model = params['model']  # copy of model setup so we aren't actually training the model stored in this dict

    conf_mat, actual, predicted, score, fold_metrics, iteration_metrics = do_cv(
        model, params['X_data'], params['y_data'], quiet=True, probability=True)

    model_prediction[model_name] = {
        'actual': actual,
        'predicted': predicted,
        'score': score
    }
    model_metrics[model_name] = iteration_metrics

    metrics = {}
    # calculate metrics for target scoring measure
    for metric_key in get_metric_dict().keys():
        metrics[metric_key + '_mean'] = np.mean(iteration_metrics[metric_key])
        metrics[metric_key + '_sd'] = np.std(iteration_metrics[metric_key])

    model_performance.append(metrics)
    metrics['Model'] = model_name
    save_name = model_name + ' Confusion Matrix'

    print(metrics)
    display_cm(conf_mat, fig_name=save_name)

performance_df = pd.DataFrame(model_performance)
cols = ['Model'] + [c for c in performance_df.columns if c != 'Model']
performance_df = performance_df[cols]
performance_df.set_index('Model', inplace=True)
performance_df = performance_df.sort_values(by='Model')
performance_df.to_csv(fig_save_dir + 'test_performance' + '_' + edss_group_type + '.csv')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(performance_df)


######################################################################
# Statistical tests functions
#
######################################################################

def repkfold_ttest(d, n1, n2):
    """
    Compute correlated t-statistic and p-value for repeated k-fold cross-validated results.
    Args:
        d (array): differences
        n1: train set size
        n2: test set size

    Returns:
        dataframe: Pandas dataframe containing the test statistic and the p-value.
    """

    n = len(d)
    corrected_var = np.var(d, ddof=1, keepdims=True) * ((1 / n) + (n2 / n1))
    corrected_std = np.sqrt(corrected_var)
    statistic = np.mean(d) / corrected_std  # Calculate t-statistic

    df = n - 1
    p_value = t.sf(np.abs(statistic), df)  # p-value for one tail

    stat_df = pd.DataFrame({"corrected_std": [corrected_std], "statistic": [statistic], "p_value": [p_value]})
    return stat_df


def plot_rocs(rocs, names="Model", plot_ci=True, ci_type="sensitivity", annotate_best=True, save_to_file=None):
    # Plot multi-model comparison with Confidence Interval shading and Best Point annotations
    fig, ax = plt.subplots(figsize=(7, 7))
    pauc.plot_roc(
        rocs,
        ax=ax,
        title=names + " ROC comparison" + (" with 95% confidence intervals" if plot_ci else ""),
        plot_ci=plot_ci,  # Shades the sensitivity CI bounds
        ci_type=ci_type,
        annotate_best=annotate_best
    )
    if not save_to_file is None:
        plt.savefig(save_to_file)
    plt.show()


test_size = int(len(X_data) * 1 / n_splits)
train_size = int(len(X_data) - test_size)


def ttest(metric_name, model1_name, model2_name, model1_metric, model2_metric):
    ttest_result = ttest_rel(model1_metric, model2_metric)
    confidence_interval = ttest_result.confidence_interval()

    print(
        f't-test: {metric_name}\t{model1_name}\t{model2_name}\t{ttest_result.statistic:.6}\t{ttest_result.p_value:.6}')
    return ttest_result.statistic, ttest_result.p_value


def corrected_ttest(metric_name, model1_name, model2_name, model1_metric, model2_metric):
    differences = [y - x for y, x in zip(model1_metric, model2_metric)]

    stat_df = repkfold_ttest(differences, train_size, test_size)
    # corrected_std = stat_df.corrected_std[0][0]
    statistic = stat_df.statistic[0][0]
    p_value = stat_df.p_value[0][0]

    print(f'corrected t-test: {metric_name}\t{model1_name}\t{model2_name}\t{statistic:.6}\t{p_value:.6}')
    return statistic, p_value


def mc_nemar_test(model1_name, model2_name, predicted1, predicted2):
    tab = pd.crosstab(predicted1, predicted2)
    # print(tab)
    mcnemar = sm.stats.mcnemar(tab, exact=False)
    print(f'mcnemar: \t{model1_name}\t{model2_name}\t{mcnemar.statistic:.6}\t{mcnemar.pvalue:.6}')
    return mcnemar.statistic, mcnemar.pvalue


def median_index(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


def delong_median_test(model1_name, model2_name, rocs1, rocs2):
    if not len(rocs1) == len(rocs2):
        raise ValueError(f"rocs1 (len={len(rocs1)}) and rocs2 (len={len(rocs2)}) must have same length")
    delongs = []
    for i in range(len(rocs1)):
        roc1 = rocs1[i]
        roc2 = rocs2[i]
        delongs.append(pauc.compare(roc1, roc2, method="delong", paired=True))

    p_values = [d.p_value for d in delongs]
    delong = delongs[median_index(p_values)]
    print(f'delong median: \t{model1_name}\t{model2_name}\t{delong.stat:.6}\t{delong.p_value:.6}')
    # print(', '.join(['{:.4}'.format(p_value) for p_value in p_values]))
    # print(f'delong\t{model1_name}\t{model2_name}\t{delong.stat:.6}\t{delong.p_value:.6}\t{delong.conf_int[0]:.6}\t{delong.conf_int[1]:.6}')
    # plot_rocs([roc1, roc2], model1_name + ' ' + model2_name)
    return delong.stat, delong.p_value


def display_stat_test_heatmap(data, title=None, save_to_file=None):
    plt.figure(figsize=(16, 8))
    with sns.plotting_context(context="paper", font_scale=3):
        heatmap = sns.heatmap(
            data,
            fmt='',
            cmap='RdBu',
            center=0.05,
            vmin=0.0,
            vmax=0.2,
            cbar_kws={'label': ''},
            linewidths=.5,
            linecolor='whitesmoke'
        )
        heatmap.set_xlabel('')
        heatmap.set_ylabel('')
        heatmap.set_facecolor('white')

        if not title is None:
            plt.title(title)
        if not save_to_file is None:
            plt.savefig(save_to_file)
        plt.show()


######################################################################
# Statistical tests to compare performance measures
#
######################################################################

# for each metric and each ML model output t-test
metric_names = get_metric_dict().keys()

results = {
    'model1': [],
    'model2': [],
    'ttest': [],
    'mcnemar': [],
    'delong': []
}


def get_iteration_rocs(model_name, actual, score):
    rocs = []
    for iteration in range(n_repeats):
        actual_iteration = actual.xs(iteration, level=2, axis=0)
        score_iteration = score.xs(iteration, level=2, axis=0)
        roc = pauc.ROC(actual_iteration, score_iteration, name=model_name)
        rocs.append(roc)
    return rocs


for m1 in range(len(model_names_ordered)):
    model1_name = model_names_ordered[m1]
    model1_metrics = model_metrics[model1_name]
    model1_prediction = model_prediction[model1_name]
    actual1 = model1_prediction['actual']
    predicted1 = model1_prediction['predicted']
    score1 = model1_prediction['score']
    actual1_id = actual1.groupby(level=[0]).median().astype(int)
    predicted1_id = predicted1.groupby(level=[0]).median().astype(int)
    score1_id = score1.groupby(level=[0]).mean()

    roc1 = pauc.ROC(actual1_id, score1_id, name=model1_name)
    rocs1 = get_iteration_rocs(model1_name, actual1, score1)

    for m2 in range(m1 + 1, len(model_names_ordered)):
        model2_name = model_names_ordered[m2]
        model2_metrics = model_metrics[model2_name]
        model2_prediction = model_prediction[model2_name]
        actual2 = model2_prediction['actual']
        predicted2 = model2_prediction['predicted']
        score2 = model2_prediction['score']
        actual2_id = actual2.groupby(level=[0]).median().astype(int)
        predicted2_id = predicted2.groupby(level=[0]).median().astype(int)
        score2_id = score2.groupby(level=[0]).mean()
        # Indexes should be identical but contain duplicated values for each data point
        assert predicted1.index.identical(predicted2.index)
        assert predicted1_id.index.identical(predicted2_id.index)

        ttest = corrected_ttest(scoring, model1_name, model2_name,
                                model1_metrics[scoring], model2_metrics[scoring])
        mcnemar = mc_nemar_test(model1_name, model2_name, predicted1_id, predicted2_id)

        rocs2 = get_iteration_rocs(model2_name, actual2, score2)
        delong_median = delong_median_test(model1_name, model2_name, rocs1, rocs2)

        roc2 = pauc.ROC(actual2_id, score2_id, name=model2_name)
        delong_pooled = pauc.compare(roc1, roc2, method="delong", paired=True)

        results['model1'].append(model1_name)
        results['model2'].append(model2_name)
        results['ttest'].append(ttest[1])
        results['mcnemar'].append(mcnemar[1])
        results['delong'].append(delong_pooled.p_value)

results_df = pd.DataFrame(results)
results_df['model1'] = pd.Categorical(results_df['model1'], model_names_ordered)
results_df['model2'] = pd.Categorical(results_df['model2'], model_names_ordered)
results_df.sort_values(by=['model1', 'model2'], ascending=True, inplace=True)
print(results_df)

ttest_pivot = results_df.pivot_table(index='model1', columns='model2', values='ttest')
display_stat_test_heatmap(ttest_pivot,
                          # title = 'Model comparison of balanced accuracy using corrected t-test p-value',
                          save_to_file=f'../output/stat_test_ttest_{edss_group_type}.png')
mcnemar_pivot = results_df.pivot_table(index='model1', columns='model2', values='mcnemar')
display_stat_test_heatmap(mcnemar_pivot,
                          # title = 'Model comparison of predictions using McNemar test p-value',
                          save_to_file=f'../output/stat_test_mcnemar_{edss_group_type}.png')
delong_pivot = results_df.pivot_table(index='model1', columns='model2', values='delong')
display_stat_test_heatmap(delong_pivot,
                          # title = 'Model comparison of AUC using DeLong test p-value',
                          save_to_file=f'../output/stat_test_delong_{edss_group_type}.png')



# un-normalise the data and try to find out which samples were misclassified
best_model_name = performance_df.sort_values(by=scoring + '_mean', ascending=False).iloc[0].name
print(best_model_name)
best_model = model_data_dict[best_model_name]

conf_mat, actual, predicted, score, fold_metrics, iteration_metrics = do_cv(
    best_model['model'], best_model['X_data'], best_model['y_data'],
    quiet=True, probability=True)

# This returns the results for all (n_repeat * n_splits) folds
# These need to be aggregated to produce a single result for each participant or progression
# actual_agg = actual.groupby(level=[0]).median()
# predicted_agg = predicted.groupby(level=[0]).median()
# score_agg = score.groupby(level=[0]).mean()

predictions = pd.DataFrame({
    'progression': actual,
    'prediction': predicted,
    'score': score
})
# predictions.index.names = ['index']

#predictions_to_be_appended = indexed_test_predictions
predictions['misclassified'] = abs(predictions['prediction'] - predictions['progression'])
predictions['correct'] = 1 - predictions['misclassified']
# predictions = predictions.drop(['progression'], axis=1)

# original_data_with_predictions = labels_X_all.join(predictions)
#
# display(original_data_with_predictions[original_data_with_predictions['misclassified'] == 1].head(5))
# original_data_with_predictions.to_csv(fig_save_dir + 'predictions_' + edss_group_type + '.csv')
predictions.to_csv(fig_save_dir + 'predictions_' + edss_group_type + '.csv')

print(best_model['X_data'])
best_model['X_data'].to_csv(fig_save_dir + 'best_model_features' + '_' + edss_group_type + '.csv')



def plot_misclassification_matrix(data_with_predictions, fig_name):
    scores = sorted(list(set(data_with_predictions['edfscr1l']).union(
        set(data_with_predictions['next_edss']))))

    if n_repeats == 1:
        error_matrix = data_with_predictions.pivot_table(
            index='edfscr1l',
            columns='next_edss',
            values='misclassified',
            aggfunc='sum',
            fill_value=0
        ).reindex(index=scores, columns=scores, fill_value=0)
        total_matrix = data_with_predictions.pivot_table(
            index='edfscr1l',
            columns='next_edss',
            values='prediction',  # Any column works here for counting
            aggfunc='count',
            fill_value=0
        ).reindex(index=scores, columns=scores, fill_value=0)
    else:
        # this hack estimates 'actual' predictions given the repeated experiments
        # Using the lambda functions in pivot_table aggfunc failed :(
        error_matrix = (data_with_predictions
                        .groupby(['edfscr1l', 'next_edss'])['misclassified']
                        .agg(lambda x: round(sum([i / n_repeats for i in x])))
                        .fillna(0).unstack(['next_edss']))
        total_matrix = (data_with_predictions
                        .groupby(['edfscr1l', 'next_edss'])['prediction']
                        .agg(lambda x: round(len(x) / n_repeats))
                        .fillna(0).unstack(['next_edss']))

    percentage_matrix = (error_matrix / total_matrix) * 100
    percentage_matrix = percentage_matrix.fillna(0)  # Clean up NaNs where total was 0

    annot_labels = np.array([
        [f'{int(count)}\\{int(total)}\n({pct:.1f}%)' if total > 0 else ""
         for count, pct, total in zip(count_row, pct_row, total_row)]
        for count_row, pct_row, total_row in zip(error_matrix.values, percentage_matrix.values, total_matrix.values)
    ])

    mask = (total_matrix == 0)

    plt.figure(figsize=(16, 8))

    sns.heatmap(
        percentage_matrix,
        annot=annot_labels,
        fmt='',
        mask=mask,
        cmap='YlOrRd',
        cbar_kws={'label': 'Percentage Misclassified (%)'},
        linewidths=.5,
        linecolor='whitesmoke'
    )

    plt.title('EDSS Transition Misclassifications ' + fig_name)
    plt.xlabel('Next EDSS')
    plt.ylabel('Previous EDSS')
    plt.show()


# read in csv files of predictions and combine into confusion matrices
predictions = {}
index_cols = ['id', 'fold', 'iteration']
for group in edss_group_names:
    predictions[group] = pd.read_csv(f'{fig_save_dir}predictions_{group}.csv')
    # for some reason so process causes the confusion_matrix function below to fail due to mixed types if the type is a float, therefore convert to an int
    predictions[group]["progression"] = predictions[group]["progression"].astype(int)
    predictions[group]["prediction"] = predictions[group]["prediction"].astype(int)

predictions_combined = pd.concat([predictions['G1'], predictions['G2'], predictions['G3']])
predictions["G0"].set_index(index_cols, inplace=True)
predictions_combined.set_index(index_cols, inplace=True)

datasets = {
    "G0": predictions['G0'],
    "G1-3": predictions_combined
}

cm_all = confusion_matrix(predictions['G0']["progression"], predictions['G0']["prediction"])
cm_combined = confusion_matrix(predictions_combined["progression"], predictions_combined["prediction"])

results = {}
for name, df in datasets.items():
    y_true = df["progression"]
    y_pred = df["prediction"]
    y_score = df["score"]

    results[name] = {
        "sensitivity": sensitivity_score(y_true, y_pred),
        "specificity": specificity_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro'),
        "recall": recall_score(y_true, y_pred, average='macro'),
        "f1": f1_score(y_true, y_pred, average='macro'),
        "auc": roc_auc_score(y_true, y_score, average='macro')
    }

df = pd.DataFrame.from_dict(results, orient='index')
print(df)
display_cm(cm_all, fig_name="Single Model Confusion Matrix", class_names=["No Progression", "Progression"])
display_cm(cm_combined, fig_name="Multiple Models Confusion Matrix", class_names=["No Progression", "Progression"])


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


rocs = []
colors = ['blue', 'red', 'green', 'yellow', 'orange', 'magenta', 'cyan']

for index, (name, df) in enumerate(datasets.items()):
    color = colors[index % len(colors)]
    light_color = adjust_lightness(color, 0.7)
    y_true = df["progression"]
    y_pred = df["prediction"]
    y_score = df["score"]

    # Plot the ROC curve for each iteration
    y_true_iter_id = y_true.groupby(level=[0, 2]).median().astype(int)
    y_score_iter_id = y_score.groupby(level=[0, 2]).mean()
    y_true_iter = y_true_iter_id.groupby(level=[1])
    y_score_iter = y_score_iter_id.groupby(level=[1])
    y_true_groups = [group for name, group in y_true_iter]
    y_score_groups = [group for name, group in y_score_iter]
    for i in range(len(y_true_groups)):
        y_true_group = y_true_groups[i]
        y_score_group = y_score_groups[i]
        fpr, tpr, thresholds = roc_curve(y_true_group, y_score_group)
        plt.plot(fpr, tpr, color=color, alpha=0.3, linewidth=1, label='')

    # Pool the results for all iterations and plot and report AUC
    y_true_id = y_true.groupby(level=[0]).median().astype(int)
    y_score_id = y_score.groupby(level=[0]).mean()

    auc_score = roc_auc_score(y_true_id, y_score_id, average='macro')
    fpr, tpr, thresholds = roc_curve(y_true_id, y_score_id)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=2.5, label=name + ' model, AUC = %0.3f' % auc_score)
    rocs.append(pauc.ROC(y_true_id, y_score_id, name=name))

plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(f'../output/roc_curve_iterations_G0_vs_G1-3.png')
plt.show()

plot_rocs(rocs, save_to_file='../output/roc_curve_with_intervals_G0_vs_G1-3.png')

delong = pauc.compare(rocs[0], rocs[1], method="delong", paired=False)
print(delong)

# plot_misclassification_matrix(predictions_all, "G0")
# plot_misclassification_matrix(predictions_combined, "G1-3")