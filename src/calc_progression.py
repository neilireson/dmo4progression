import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt

import parameters as params
import load_data as data

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
    for name in params.edss_group_scores.keys():
        combined.append(demographics_table(
            data[data[params.participant_id_var]
            .isin(edss_group_participant_data[name][params.participant_id_var])]))

    combined_df = pd.DataFrame(combined).transpose()
    combined_df.columns = list(params.edss_group_scores.keys())
    print(combined_df.to_latex())




#####################################################
# Functions to calculate participant-based progression
#
#####################################################

timepoints = ['T1', 'T2', 'T3', 'T4', 'T5']


def calc_progression(tp_scores, min_score, return_all=False, select=params.progression_select):
    progressions = []
    non_progressions = []

    for i, tp0 in enumerate(timepoints):
        if i == len(tp_scores) - params.min_progression_points:
            break
        if tp_scores[tp0] >= min_score:
            valid_future_points = []
            for j in range(i + 1, len(timepoints)):
                if not np.isnan(tp_scores.iloc[j]):
                    valid_future_points.append(tp_scores.iloc[j])
            # if there are at least min_progression_points future non-null timepoints, then this is valid
            if len(valid_future_points) >= params.min_progression_points:
                min_progression = params.min_increase_for_progression(tp_scores[tp0])
                is_progression = True
                # if any of the consecutive future timepoint scores is
                # less than required for progression, this is not progression
                for j in range(params.min_progression_points):
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
    feature_file_path = params.data_dir + f'/{timepoint}{params.dmo_data_dir_suffix}/cvs-{timepoint}{params.dmo_data_file_suffix}'
    temp_df = pd.read_csv(feature_file_path)
    dfs_list.append(temp_df)

dmo_data = pd.concat(dfs_list, ignore_index=True)
# convert all column names to lower case for consistency between versions
dmo_data.columns = dmo_data.columns.str.lower()
dmo_data[params.dmo_visit_var] = dmo_data[params.dmo_visit_var].str.upper()  # for consistency with filenames

merged_data = pd.merge(data.data, dmo_data, left_on=[params.participant_id_var, 'visit.number'],
                       right_on=[params.dmo_participant_id_var, params.dmo_visit_var], how='left')

if not params.dmo_participant_id_var == params.participant_id_var:
    merged_data = merged_data.drop([params.dmo_participant_id_var], axis=1)
merged_data = merged_data.drop([params.dmo_visit_var], axis=1)

# print(f'Column NaN counts out of {len(merged_data)} T1, T2, T3 rows')
# print(merged_data[merged_data['visit.number'].isin({'T1', 'T2', 'T3'})].isna().sum())

# drop T1, T2, and T3 rows if they contain nan, which will be due to missing DMOs
count_before = merged_data[params.participant_id_var].nunique()
print(f'{count_before} participants read from file...')

# get indices of rows with any nan values
nan_indices = merged_data.index[merged_data.isna().any(axis=1)]
# get indices of rows with T1, T2 or T3 values
timepoint_indices = merged_data.index[merged_data['visit.number'].isin({'T1', 'T2', 'T3'})]
# Get intersection of T1, T2, T3 rows with rows containing any nan
intersection = nan_indices.intersection(timepoint_indices)
merged_data = merged_data.drop(intersection)
print(
    f'{count_before - merged_data[params.participant_id_var].nunique()} participants excluded with missing DMOs at timepoints T1, T2 and T3.')

participant_data = (merged_data[[params.participant_id_var, 'visit.number', 'edfscr1l']]
                    .pivot(index=params.participant_id_var, columns='visit.number', values='edfscr1l'))

count_before = len(participant_data)
participant_data = participant_data[participant_data[['T1', 'T2', 'T3']].isna().sum(axis='columns') < 3]
print(
    f'{count_before - len(participant_data)} participants excluded as have T1-3 EDSS scores are all NaN...')

base_scores = params.edss_group_scores['G0']

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
                                columns=[params.participant_id_var, 'timepoint', 'next_edss', 'progression'])

print('Participants excluded as they have no valid timepoints')
print(participant_data[participant_data['timepoint'].isna()])
# remove all participants who have insufficient (i.e. <3) timepoints
count_before = len(participant_data)
participant_data = participant_data.dropna(subset=['timepoint'])
print(f'{count_before - len(participant_data)} participants excluded who have insufficient (i.e. <3) timepoints...\n')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(participant_data[participant_data['progression'] == 1].head(100))

participant_data = participant_data.merge(merged_data, how='left',
                                          left_on=[params.participant_id_var, 'timepoint'],
                                          right_on=[params.participant_id_var, 'visit.number'])
# remove processing columns
participant_data = participant_data.drop(['T1', 'T2', 'T3', 'T4', 'T5', 'timepoint'], axis=1)

progression_data = progression_data.merge(merged_data, how='left',
                                          left_on=[params.participant_id_var, 'timepoint'],
                                          right_on=[params.participant_id_var, 'visit.number'])
# remove processing column
progression_data = progression_data.drop(['timepoint'], axis=1)

edss_group_participant_data = {}
for group in params.edss_group_scores.keys():
    edss_group_participant_data[group] = participant_data[participant_data['edfscr1l'].isin(params.edss_group_scores[group])]
    print(f'EDSS group {group} size: {len(edss_group_participant_data[group])}')

get_characteristic_table(edss_group_participant_data, data.all_data)

if not params.edss_group_type == 'G0':
    print(f'Selecting EDSS scores for group {params.edss_group_type}: {params.edss_group_scores[params.edss_group_type]}')
    count_before = len(participant_data)
    participant_data = edss_group_participant_data[params.edss_group_type]
    print(f'Participants data: {count_before - len(participant_data)} participants excluded who are not in the group')

    participants_before = progression_data[params.participant_id_var].nunique()
    progressions_before = len(progression_data)
    progression_data = progression_data[progression_data['edfscr1l'].isin(params.edss_group_scores[params.edss_group_type])]
    print(
        f'Progression data: {participants_before - progression_data[params.participant_id_var].nunique()} participants excluded who are not in the group')
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
plt.savefig('../output/progressors_' + params.edss_group_type + '.png')
plt.show()

#print(participant_data[params.participant_id_var].unique())
