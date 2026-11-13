############################################################
# Calculate EDSS score consecutive timepoint transitions
#
############################################################

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import parameters as params

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
    plt.savefig('../output/' + name + '_' + params.edss_group_type + '.png')
    plt.show()

base_scores = params.edss_group_scores['G0']

def progression(row):
    return 1 if ((row['edfscr1l'] > base_scores[0]) &
                 (row['next_edss'] >= base_scores[0]) &
                 ((row['next_edss'] - row['edfscr1l']) >= params.min_increase_for_progression(row['edfscr1l']))) else 0


def get_transitions(data):
    transition_data = data.copy()
    print(f'{transition_data[params.participant_id_var].nunique()} participants')
    print(f'{len(transition_data)} total participant timepoints rows')

    # shift timepoints and EDSS to get next as columns
    transition_data['next_edss'] = transition_data.groupby(params.participant_id_var)['edfscr1l'].shift(-1)
    transition_data['Next_Timepoint'] = transition_data.groupby(params.participant_id_var)['visit.number'].shift(-1)

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

    # get progression label column
    transition_data['progression'] = transition_data.apply(progression, axis=1)

    # transitions = transition_data[progression_mask]  # all the data where a transition happens, just in case.
    #print(transitions.head(10))

    # now that we have a reduced dataset, loop through our weekly dmo aggregations and append the relevant ones to the dataframe

    dfs_list = []
    # we don't need DMOs where T5 is previous
    for timepoint in ['T1', 'T2', 'T3', 'T4']:
        feature_file_path = params.data_dir + f'/{timepoint}{params.dmo_data_dir_suffix}/cvs-{timepoint}{params.dmo_data_file_suffix}'
        temp_df = pd.read_csv(feature_file_path)
        dfs_list.append(temp_df)

    dmo_data = pd.concat(dfs_list, ignore_index=True)
    # convert all column names to lower case for consistency between versions
    dmo_data.columns = dmo_data.columns.str.lower()
    dmo_data[params.dmo_visit_var] = dmo_data[params.dmo_visit_var].str.upper()  # for consistency with filenames

    transition_data = pd.merge(transition_data, dmo_data, left_on=[params.participant_id_var, 'visit.number'],
                               right_on=[params.dmo_participant_id_var, params.dmo_visit_var], how='left')
    if not params.dmo_participant_id_var == params.participant_id_var:
        transition_data = transition_data.drop([params.dmo_participant_id_var], axis=1)
    transition_data = transition_data.drop([params.dmo_visit_var], axis=1)

    # drop all remaining NAN rows, which will be due to missing DMOs
    count_before = len(transition_data)
    transition_data = transition_data.dropna()
    print(f'{count_before - len(transition_data)} rows have missing DMOs and will be dropped...')

    num_progressions = len(transition_data[transition_data['progression'] == 1])
    unique_sub_progressions = len(transition_data[transition_data['progression'] == 1][params.participant_id_var].unique())
    total_subs = transition_data[params.participant_id_var].nunique()

    print(
        f'We have {num_progressions} progressions across {unique_sub_progressions} subjects, out of a total of {len(transition_data)} differences between timepoints across {total_subs} subjects.')

    print(transition_data[transition_data['progression'] == 1].head(5))

    show_transition_matrix(transition_data, 'trimmed_transitions')

    # drop processing columns
    transition_data = transition_data.drop(['next_edss', 'Next_Timepoint'], axis=1)

    return transition_data
