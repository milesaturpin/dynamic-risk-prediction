
from utils import generate_col_dict
from train_test_split import train_val_test
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from time import time
from scipy.stats import skew, linregress
import re

proc_path = 'data/processed/'

encounters = pd.read_pickle(proc_path + 'encounters.pickle')
outcomes   = pd.read_pickle(proc_path + 'outcomes.pickle')

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
    EDIT: added sbo poa then dropped so could filter
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

def standardize_times(df):
    '''
    - Standardize times to time of admission
        - Aggregate by 1,4,6 hours??
    - surg datetime hour?
    - Add time to event column
    '''
    start = time()
    df['min_datetime'] = (df.datetime
                          .groupby(level=[0,1])
                          .transform(lambda x: x.min()))

    to_hour = lambda td: td.total_seconds() // 3600
    df['hour_since_adm'] = ((df.datetime - df.min_datetime)
                            .transform(to_hour))
    df['surg_datetime_hour_enc'] = ((df.surg_datetime_enc - df.min_datetime)
                                .transform(to_hour))
    df = (df
          .pipe(filter_populations, 48)
          .drop(['datetime', 'min_datetime'], 1)
          .reset_index()
          [lambda df: df.hour_since_adm <= 48]
          .groupby(['mrn', 'id','hour_since_adm'])
          .agg('mean')
          .sort_index(level=[0,1,2])
          .reset_index(level=2)
          .assign(hour_since_adm = lambda df: df.hour_since_adm.astype(int))
          .set_index('hour_since_adm', append=True))

    print(f'Finished standardizing times in {round(time()-start)}s')
    return df

def filter_populations(df, window):
    """
    Filter out those who go to surgery or are discharged
    before time t. Trying 48 hours for now.

    TODO: create time_to_event column and filter out those
    with < window
    """
    print(df.reset_index()['id'].nunique())

    time_to_event = (df.reset_index()
                     .groupby('id')
                     .apply(lambda s: s.hour_since_adm.max() > window)
                     )
    print(time_to_event.value_counts())

    df = df.loc[pd.IndexSlice[:,list(time_to_event[time_to_event].index),:],:]

    print(df.reset_index()['id'].nunique())

    return df

def fix_occurrences(df):
    col_dict = generate_col_dict(df)
    df1 = df[col_dict['occ']]
    df1[df1 != 0] = 1
    df.loc[:, col_dict['occ']] = df1
    return df

def fill_na(df):
    """
    Fill in na's for those who have no recorded measurement
    within 48 hours. If this is common maybe add an indicator
    variable denoting missing.
    """
    return df

def linear_model(s):
    """
    Fit a linear model to numerical measurements and
    poisson process rates.
    """
    '''print('fix this')
                if len(s) < 2:
                    return 0, 0, 0

                x = s.index
                y = s.values
                slope, intercept, r_value, p_value, std_err = linregress(x,y)'''

    #return slope, r_value, std_err
    return s.iloc[-1] - s.iloc[0]

def group_stats(df, fns, fn_names):
    """

    """

    df = (df.agg(fns).reset_index())
    # name agg functions
    df['index'] = pd.Series(fn_names)
    # make new names the index
    df = df.set_index('index').unstack().reset_index()

    df['joined'] = df['index']+'_'+df['level_0']
    df = df.set_index('joined')[0].rename('value')

    return df


def summary_stats(df):
    """
    Calculate summary stats for each measurement
    of patients.
    """

    col_dict = generate_col_dict(df)
    df.iloc[:2,:] = df.iloc[:2,:].fillna(0)

    enc = df[col_dict['enc']].tail(1).squeeze()

    # Custom agg functions
    def rolling_sum(s):
        s = s.reset_index(level=[0,1],drop=True).reindex(range(48))
        s_rolling = s.rolling(6,min_periods=0).sum()
        #TODO: change to linreg
        slope = linear_model(s_rolling)
        return slope

    last = lambda s: s.iloc[-1]
    tsl = lambda s: 48 - max(s.reset_index().hour_since_adm)

    # Agg
    numeric_stats = [last,  tsl,  'mean', 'std',
                     'min', 'max', 'skew', 'count']
    numeric_stat_names = ['last',  'tsl',  'mean', 'std',
                     'min', 'max', 'skew', 'count']
    occ_stats = [last,  tsl, rolling_sum, 'count']
    occ_stat_names = ['last',  'tsl', 'rolling_sum', 'count']

    numeric_summ = group_stats(df[col_dict['vitals'] + col_dict['labs']],
                               numeric_stats, numeric_stat_names)
    occ_summ = group_stats(df[col_dict['io'] + col_dict['occ']],
     occ_stats, occ_stat_names)

    # TODO: make own rate thing
    return pd.concat([enc, numeric_summ, occ_summ])

def scale(df):
    """
    Normalize values.
    - Scale before or after summary stats?
    """

    return df


def preprocess(sbo, window=48):
    """
    Other columns
    - age
    """
    start = time()

    sbo48 = (sbo
             .pipe(merge_encounters)
             .pipe(standardize_times)
             .pipe(fix_occurrences)
             .groupby(level=[0,1])
             .apply(summary_stats)
             )
    print(sbo48.index)

    seed = 104

    train, val, idx_dict = train_val_test(sbo48, seed)

    # write out preprocessed dataframe for visualization
    sbo48.to_pickle('data/processed/sbo48.pickle')



    #y = sbo48.any_sbo_surg
    #x, x_cols = (sbo48.pipe(scale))
    print(f'Finished processing in {round(time()-start)}')

    return sbo48

if __name__ == '__main__':
    sbo = pd.read_pickle('data/processed/sbo.pickle')
    sbo = preprocess(sbo)
    print(sbo.shape)
    print(sbo.reset_index()['id'].nunique())
    print(sbo.reset_index().columns)
    sbo.to_pickle('data/processed/sbo48.pickle')















