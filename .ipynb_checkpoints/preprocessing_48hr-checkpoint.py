
from utils import generate_suffix_dict, generate_prefix_dict
from train_test_split import train_val_test
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from time import time
from scipy.stats import linregress
import os

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
    col_dict = generate_suffix_dict(df)
    df.loc[:, col_dict['occ']] = (df[col_dict['occ']]
                                  .fillna(0)
                                  .astype(bool)
                                  .astype(int))
    return df

def linear_model(s):
    """
    Fit a linear model to numerical measurements and
    poisson process rates.
    """

    x = s.index
    y = s.values
    slope, intercept, r_value, p_value, std_err = linregress(x,y)

    return slope, r_value, std_err
    #return s.iloc[-1] - s.iloc[0]

def group_stats(df, fns):
    """

    """
    df = df.agg(fns).unstack().rename('value').reset_index()
    df['joined'] = df['level_1']+'_'+df['level_0']
    df = df.set_index('joined')['value']

    return df


def summary_stats(df):
    """
    Calculate summary stats for each measurement
    of patients.
    """

    col_dict = generate_suffix_dict(df)
    #df.iloc[:2,:] = df.iloc[:2,:].fillna(0)

    enc = df[col_dict['enc']].tail(1).squeeze()

    # Custom agg functions
    def slope_numeric(s):
        s = s.reset_index(level=[0,1],drop=True)
        slope, r_value, std_err = linear_model(s)
        return slope

    def slope_rolling_sum(s):
        s = s.reset_index(level=[0,1],drop=True).reindex(range(48))
        s_rolling = s.rolling(6,min_periods=0).sum()
        #TODO: change to linreg
        #slope = linear_model(s_rolling)

        slope, r_value, std_err = linear_model(s_rolling)
        return slope

    def last(s):
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return s.iloc[-1]

    def tsl(s):
        s = s.dropna()
        if len(s) == 0:
            return 48
        return 48 - max(s.reset_index().hour_since_adm)

    def tsl_occ(s):
        s = s.reset_index(level=[0,1],drop=True).dropna()
        if len(s) == 0:
            return 48
        nonzero = s[s>0].index
        if len(nonzero) == 0:
            return 48
        if max(nonzero) == max(nonzero):
            return 48 - max(nonzero)
        else:
            return 48

    numeric_stats = [last,  tsl, slope_numeric, 'mean', 'std',
                     'min', 'max', 'skew']
    io_stats      = [last, tsl_occ, slope_rolling_sum, 'sum']
    occ_stats     = [tsl_occ, slope_rolling_sum, 'sum']

    numeric_summ = group_stats(df[col_dict['vitals'] +
                                  col_dict['labs']], numeric_stats)
    io_summ      = group_stats(df[col_dict['io']], io_stats)
    occ_summ     = group_stats(df[col_dict['occ']], occ_stats)

    return pd.concat([enc, numeric_summ,io_summ, occ_summ])

def fill_na(df):
    """
    Fill in na's for those who have no recorded measurement
    within 48 hours. If this is common maybe add an indicator
    variable denoting missing.

    - slope = 0
    - tsl = 48
    - last = mean
    - min = mean
    - max = mean
    - std = mean
    - skew = mean
    - mean = mean
    - sum = 0
    """
    prefix_dict = generate_prefix_dict(list(df.columns))

    lst, tsl, slp, men, std, mn, mx, sm, skw = (prefix_dict[i] for i in ['last', 'tsl', 'slope',
                                                  'mean', 'std', 'min', 'max',
                                                  'sum', 'skew'])
    df.loc[:, slp + sm] = df[slp + sm].fillna(0)

    df.loc[:, tsl] = df[tsl].fillna(48)

    df.loc[:,lst + men + std + mn + mx + skw] = (df[lst+men+std+mn+mx+skw]
                                                 .pipe(lambda df: df.fillna(df.mean())))

    return df


def scale(df):
    """
    Normalize values.
    - Scale before or after summary stats?
    """

    return df


def preprocess(sbo, window=48):
    """
    Other columns

    """
    start = time()

    if not os.path.exists('data/processed/sbo48.pickle'):
        sbo48 = (sbo
                 .pipe(merge_encounters)
                 .pipe(standardize_times)
                 .pipe(fix_occurrences)
                 .groupby(level=[0,1])
                 .apply(summary_stats)
                 )
        seed = 104

        train, val, idx_dict = train_val_test(sbo48, seed)

        train48 = fill_na(train)
        val48   = fill_na(val)

        # write out preprocessed dataframe for visualization
        sbo48.to_pickle('data/processed/sbo48.pickle')
        train48.to_pickle('data/processed/train48.pickle')
        val48.to_pickle('data/processed/val48.pickle')
    else:
        print('Reading in file')
        train48 = pd.read_pickle('data/processed/train48.pickle')
        val48 = pd.read_pickle('data/processed/val48.pickle')

    x_train = train48.loc[:, 'last_bp_dia_vitals':].fillna(0).values
    y_train = train48.any_sbo_surg_enc.values
    x_val = val48.loc[:, 'last_bp_dia_vitals':].fillna(0).values
    y_val = val48.any_sbo_surg_enc.values

    print(f'Finished processing in {round(time()-start)}')

    return x_train, y_train, x_val, y_val

if __name__ == '__main__':
    sbo = pd.read_pickle('data/processed/sbo.pickle')
    sbo = preprocess(sbo)
    print(sbo.shape)
    print(sbo.reset_index()['id'].nunique())
    #sbo.to_pickle('data/processed/sbo48.pickle')















