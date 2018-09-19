"""
Fill in missing data, create rate columns, do train test split.
"""

import pandas as pd
import numpy as np
from time import time
from train_test_split import train_val_test
from utils import unique_surg, generate_col_dict

proc_path = 'data/processed/'

encounters = pd.read_pickle(proc_path + 'encounters.pickle')
outcomes   = pd.read_pickle(proc_path + 'outcomes.pickle')
sbo        = pd.read_pickle(proc_path + 'sbo.pickle')

start = time()
agg_hour = 1

def surg_datetimes(df):
    '''
    Add columns representing the time of an SBO surgery
    or a non-SBO surgery... used for slicing time blocks

    NOTES: currently filtering out IDs from encounters by if they appear in the sbo df
    '''
    #display(df.sort_index(level=[0,1]))
    eligible_ids = df.index.get_level_values(1).drop_duplicates()

    sbo_surg_datetimes = (encounters[['mrn', 'id', 'surg_datetime']]
                          .rename(columns={'surg_datetime': 'datetime'})
                          .assign(time_of_surg = lambda df: df.datetime.notna().astype(int))
                          .dropna()
                          # so don't get one liners
                          [lambda df: df['id'].isin(eligible_ids)]
                          .set_index(['mrn', 'id', 'datetime']))

    non_sbo_surg_datetimes = (outcomes[outcomes['non_sbo_surg']]
                              .pipe(unique_surg)
                              [['mrn', 'id', 'surg_datetime']]
                              .rename(columns={'surg_datetime': 'datetime'})
                              .assign(time_of_non_sbo_surg = lambda df: df.datetime.notna().astype(int))
                              [lambda df: df['id'].isin(eligible_ids)]
                              .set_index(['mrn', 'id', 'datetime']))
    df = (pd.concat([df,
                       sbo_surg_datetimes,
                       non_sbo_surg_datetimes])
            .sort_index(level=[0,1,2]))

    return df

def merge_encounters(df):
    '''
    Merge with encounters dataframe to get
    - admission datetime
    - whether person had surgery
    - age
    - surgery datetime
    '''

    '''
    CURRENTLY DOING INNER JOIN SO DONT HAVE ONE LINE ENTRIES
    EDIT: added sbo poa then dropped
    '''
    enc = (encounters
           .set_index(['mrn','id'])
            [['adm_datetime', 'any_sbo_surg', 'age',
             'surg_datetime', 'sbo_poa']]
            .add_suffix('_enc'))

    return (pd.merge(enc, df.reset_index(level=2),
                     left_index=True, right_index=True,
                     how='inner')
            .assign(any_sbo_surg_enc = lambda df: df.any_sbo_surg_enc.astype(int))
            [lambda df: df.sbo_poa_enc]
            .drop('sbo_poa_enc', 1))

def filter_times(df):
    '''
    Remove values taken after SBO surgery
    and before any non SBO surgery (?)
    '''
    #display(df.reset_index(level=2)[lambda df: ((df.datetime < df.surg_datetime) | df.surg_datetime.isna())])

    return (df[lambda df: ((df.datetime <= df.surg_datetime_enc)
                           | (df.surg_datetime_enc.isna()))])

def standardize_times(df):
    '''
    - Standardize times to time of admission
        - Aggregate by 1,4,6 hours??
    - surg datetime hour?
    - Add time to event column
    '''
    df['min_datetime'] = (df.datetime
                          .groupby(level=[0,1])
                          .transform(lambda x: x.min()))

    to_hour = lambda td: td.total_seconds() // (agg_hour * 3600)
    df['hour_since_adm'] = ((df.datetime - df.min_datetime)
                            .transform(to_hour))
    df['surg_datetime_hour_enc'] = ((df.surg_datetime_enc - df.min_datetime)
                                .transform(to_hour))

    df = (df[lambda df: ((df.hour_since_adm < 240) &
                          (df.any_sbo_surg_enc == False)) | df.any_sbo_surg_enc]
          .drop(['datetime', 'min_datetime'], 1)
          .reset_index()
          .groupby(['mrn', 'id','hour_since_adm'])
          .agg('mean')
          .sort_index(level=[0,1,2])
          .reset_index(level=2)
          .assign(hour_since_adm = lambda df: df.hour_since_adm.astype(int))
          .set_index('hour_since_adm', append=True)
          .groupby(level=[0,1])
          .apply(fill_missing_times))

    print(f'Finished standardizing times in {round(time()-start)}s')
    return df

def fill_missing_times(df):
    '''
    - Fill in missing hours for hour_since_adm and time_to_event
    '''

    surg_datetime = df.surg_datetime_hour_enc[0]
    idx = np.arange(df.index.get_level_values(2).min(),
                    df.index.get_level_values(2).max()+1)

    # difference betweeen surg_datetime and current time
    tte = [surg_datetime - t for t in idx]

    return (df.reset_index(level=[0,1], drop=True)
            .reindex(idx)
            .assign(time_to_event_surg = pd.Series(tte, index=idx)))

def fill_nas(df):
    '''
    Fill missing values according to protocols:
    - vitals/labs -- fill forward values; fill rest
    with reference value from column so long as
    only
    - io -- fill with zeroes
    - io_unmeasured -- fill with zeroes

    TODO: code for seeing how many values have to fill
    with reference value
    '''

    col_dict = generate_col_dict(list(df.columns))
    endings = ['enc', 'surg', 'vitals', 'labs', 'io', 'occ']
    enc, surg, vitals, labs, io, occ = (col_dict[end] for end in endings)

    df.loc[:,enc] = (df[enc]
                     .groupby(level=[0,1])
                     .fillna(method='ffill'))

    df.loc[:,surg] = (df[surg]
                      #.groupby(level=[0,1])
                      #.fillna(method='bfill', limit=4)
                      .fillna(0))

    df.loc[:, vitals+labs] = (df[vitals+labs]
                              .groupby(level=[0,1])
                              .fillna(method='ffill')
                              .pipe(lambda df: df.fillna(df.mean())))

    df.loc[:, io+occ] = (df[io+occ]
                         .groupby(level=[0,1])
                         .fillna(0))
    print(f'Filled missing data in {round(time()-start)}')
    return df

def fix_occurrences(df):
    col_dict = generate_col_dict(df)
    df1 = df[col_dict['occ']]
    df1[df1 != 0] = 1
    df.loc[:, col_dict['occ']] = df1
    return df

def add_rate_columns(df):
    '''
    TODO: Add columns for
    io
    - time since last output/intake
    io_unmeasured
    - total number of stool/emesis occurences
    - time since last occurrence
    '''
    col_dict = generate_col_dict(list(df.columns))
    new_cols = [('vitals', 'mean'), ('labs', 'mean'),
                ('io', 'sum'), ('occ', 'sum')]

    settings = [[x[0], x[1], window]
                for x in new_cols
                for window in [6,12,24]]

    def create_rate_columns(col_key, func, window):
        suffix = str(window) + func
        cols = col_dict[col_key]
        return (df[cols]
                .rolling(window, min_periods=0)
                .agg(func)
                .add_suffix(suffix))

    rates = pd.concat([create_rate_columns(*setting)
                       for setting in settings], 1)

    return pd.concat([df, rates], 1).reset_index(level=[0,1], drop=True)

if __name__ == '__main__':

    sbo = (sbo
           .pipe(surg_datetimes)
           .pipe(merge_encounters)
           .pipe(filter_times)
           .pipe(standardize_times))
    print(f'Finished standardizing dataframe in {round(time()-start)}s')

    # Perform train test split before further modification,
    # to avoid leaking information
    train, val, idx_dict = train_val_test(sbo)
    print(f'Finished train-test split in {round(time()-start)}s')

    train = (train
             .pipe(fill_nas)
             .pipe(fix_occurrences)
             .groupby(level=[0,1])
             .apply(add_rate_columns))
    print(f'Finished adding rate columns to train in {round(time()-start)}s')

    val = (val
           .pipe(fill_nas)
           .pipe(fix_occurrences)
           .groupby(level=[0,1])
           .apply(add_rate_columns))

    print(f'Finished adding rate columns to val in {round(time()-start)}s')

    train.to_pickle(proc_path + 'train.pickle')
    val.to_pickle(proc_path + 'val.pickle')
    print(f'Finished writing pickles in {round(time()-start)}s')




