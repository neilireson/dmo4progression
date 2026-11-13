#####################################################
# Load data
#
#####################################################

import pandas as pd

import parameters as params

# read in main MS dataset
data_file_path = params.data_dir + params.data_file
all_data = pd.read_csv(data_file_path)
all_data['visit.number'] = all_data['visit.number'].str.upper()  # for consistency with filenames
# convert all column names to lower case for consistency between versions
all_data.columns = all_data.columns.str.lower()

# get participant demographics
demo_data = all_data.copy()
demo_data = demo_data[demo_data['visit.number'] == 'T1']

# after getting table values, just crop to the necessary columns
cols_from_dataset = [params.participant_id_var, 'edfscr1l', 'visit.number',
                     'age', 'gender', 'mstypeonset', 'mstypediag', 'visit.date']
data = all_data[cols_from_dataset]

# forward fill within groups to ensure no leakage of demographics between participants
cols_from_dataset = [col for col in cols_from_dataset if col in ['age', 'gender', 'mstypeonset', 'mstypediag']]
for col in cols_from_dataset:
    data[col] = data.groupby(params.participant_id_var)[col].ffill()

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
count_before = data[params.participant_id_var].nunique()
print(f"Dropping {sum(data['edfscr1l'].isna())} rows with no EDSS score...")
data = data.dropna(subset=['edfscr1l'])
print(
    f'Dropping rows results in {count_before - data[params.participant_id_var].nunique()} participants excluded with no EDSS scores...')

print(f'Nan counts at each timepoint')
g = data.groupby('visit.number')
print(g.count().rsub(g.size(), axis=0))

# Crop down to just the necessary columns
data = data.sort_values(by=[params.participant_id_var, 'visit.number'])

