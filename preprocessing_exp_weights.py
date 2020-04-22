from utils import generate_suffix_dict, generate_prefix_dict, generate_midfix_dict, to_json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from time import time
from preprocess_imaging import preprocess_imaging

proc_path = 'data/processed/'

encounters = pd.read_pickle(proc_path + 'encounters.pickle')
past_sbo = pd.read_pickle(proc_path + 'past_sbo.pickle')

def merge_encounters(df):
    '''
    Resulting dataframe is (656726, 29) with index (mrn, id);
    time is not standardized.
    '''
    # Encounters contains 6500 unique encounters
    enc = (
        encounters[encounters.sbo_poa]
        [['adm_datetime', 'any_sbo_surg', 'age',
          'surg_datetime', 'location']]
        .add_suffix('_enc'))
    loc_dummies = pd.get_dummies(enc['location_enc']).rename(
        columns = {'Duke Raleigh Parent Hospital':'raleigh_loc_enc',
                   'Duke Regional Parent Hospital':'regional_loc_enc',
                   'Duke University Parent Hosp': 'duke_loc_enc'})

    enc = pd.concat([enc.drop('location_enc',1), loc_dummies], axis=1)

    enc = pd.merge(enc, past_sbo,
                   left_index=True, right_index=True,
                   how='left')
    enc['past_sbo_enc'] = enc['past_sbo_enc'].fillna(0)

    merged_df = pd.merge(
        enc, df.reset_index(level=2),
        left_index=True, right_index=True,
        how='inner'
    )
    merged_df['any_sbo_surg_enc'] = merged_df['any_sbo_surg_enc'].astype(int)

    return merged_df


def merge_imaging(df):
    """
    Call merge imaging helper function
    """
    start = time()
    print('>>> Analyzing imaging text...')
    df = preprocess_imaging(df)
    enc_cols = ['adm_datetime_enc', 'age_enc', 'any_sbo_surg_enc', 'past_sbo_enc',
       'surg_datetime_enc', 'raleigh_loc_enc', 'regional_loc_enc', 'duke_loc_enc']
    df.loc[:,enc_cols] = (
        df[enc_cols].groupby(level=[0,1]).transform(lambda s: s.max()))
    df.columns = df.columns.str.replace(' ', '_')
    #.dropna(subset=['datetime'])
    print('\nFinished analyzing imaging text in {}s'.format(round(time()-start)))
    return df


def standardize_times(df):

    start = time()
    df['min_datetime'] = (
        df.datetime
        .groupby(level=[0,1])
        .transform(lambda x: x.min())
    )
    to_hour = lambda td: td.total_seconds() // 3600
    df['hour_since_adm'] = (
        (df.datetime - df.min_datetime)
        .transform(to_hour)
        .astype(int)
    )

    df['hsa_enc'] = df['hour_since_adm']

    nonsurg_cutoff = 21*24
    df['max_datetime_hour'] = (
        df.hour_since_adm
        .groupby(level=[0,1])
        .transform(lambda x: min(nonsurg_cutoff, x.max()))
    )

    df['event_hour_enc'] = (
        (df.surg_datetime_enc - df.min_datetime)
        .transform(to_hour)
        .fillna(df['max_datetime_hour'])
    )

    df['time_to_event_enc'] = (
        df.event_hour_enc - df.hour_since_adm - 1
    )

    df['time_of_day_enc'] = (
        (df.hour_since_adm + df.min_datetime.apply(lambda dt: dt.hour)) % 24
    )

    df = filter_populations(df, surg_cutoff=21*24)

    # Filter out measurements after surgery
    df = df[df.hour_since_adm < df.event_hour_enc]

    df = df.drop(['datetime', 'min_datetime',
                  'max_datetime_hour', 'surg_datetime_enc',
                 'adm_datetime_enc', 'event_hour_enc'], 1)

    col_dict = generate_suffix_dict(df)

    mean_columns = (
        df.reset_index()
        .loc[:, col_dict['vitals'] + col_dict['labs']
        + col_dict['img']
        + ['mrn', 'id','hour_since_adm']]
        .groupby(['mrn', 'id','hour_since_adm'])
        .agg('mean')
    )

    sum_columns = (
        df.reset_index().loc[:, col_dict['io'] + col_dict['occ'] + ['mrn', 'id','hour_since_adm']]
        .groupby(['mrn', 'id','hour_since_adm'])
        .agg('sum')
    )

    enc = (
        df.reset_index()[col_dict['enc'] + ['mrn', 'id','hour_since_adm']]
        .groupby(['mrn', 'id','hour_since_adm'])
        .agg('max')
    )

    print('Finished standardizing times in {}s'.format(round(time()-start)))
    res = pd.concat([enc, mean_columns, sum_columns], axis=1)
    return res

def filter_populations(df, surg_cutoff):
    ''' '''
    print('\nBefore')
    print("Number of patients: " + str(df.reset_index().mrn.nunique()))
    print('Number of encounters: ' + str(df.reset_index()['id'].nunique()))
    print(df.reset_index()[['id', 'any_sbo_surg_enc']].drop_duplicates().any_sbo_surg_enc
                .value_counts().transform(lambda x: x/x.sum()))

    df = (df.groupby(level=[0,1])
          .filter(
              lambda df: not ((df.event_hour_enc > surg_cutoff) & (df.any_sbo_surg_enc == 1)).all()
    ))

    print('\nAfter')
    print("Number of patients: " + str(df.reset_index().mrn.nunique()))
    print('Number of  encounters: '+ str(df.reset_index()['id'].nunique()))
    print(df.reset_index()[['id', 'any_sbo_surg_enc']].drop_duplicates().any_sbo_surg_enc
            .value_counts().transform(lambda x: x/x.sum()))

    return df

def fix_occurrences(df):
    col_dict = generate_suffix_dict(df)
    df.loc[:, col_dict['occ']] = (
        df[col_dict['occ']]
        .fillna(0).astype(bool).astype(int)
    )
    return df



def create_rolling_matrix(data, window):
    """
    Return rolling matrix of shape (m, n, window).
    Pad with zeros so that we don't shrink.
    """
    n, m = data.shape
    buffer_ = np.zeros((window-1, m))
    s2 = np.concatenate([buffer_, data])
    rolling_matrix = np.empty((m, n, window))
    for i in range(n):
        start = i
        end   = i + window
        # (m,0,window) <- (window,m).T
        rolling_matrix[:,i,:] = s2[start:end].T
    return rolling_matrix

def expsum(data, zerolifes):
    """
    Vectorized form of exponentially weighted sum.
    We truncate the exponential sum when the weight
    reaches 0.05 and we parameterize by the number of
    steps to get there, the "zerolife".

    Params
    ------
    data (pd.DataFrame) -- df of size (n,m)
    zerolifes (list) -- list of length z,
        a zerolife of 3 will have a window_len of 3+1

    Return dataframe of size (n, m*z)
    """
    z = len(zerolifes)
    n, m = data.shape
    # Take max of zerolifes + 1 to get window length
    max_zerolife = max(zerolifes)
    window_len   = max(zerolifes) + 1
    rolling_matrix = create_rolling_matrix(data.values, window_len)

    # Create matrix of shape (window_len, z)
    # Buffer with 0 to accommodate diff win lengths
    expvals = np.empty((window_len, z))
    for zi, zerolife in enumerate(zerolifes):
        buffer_len = max_zerolife - zerolife
        expvals[:,zi] = np.array(
            [0] * buffer_len
            + [ 0.05**(t/zerolife) for t in np.arange(zerolife + 1) ][::-1]
        )

    # (m, n, window_len) @ (window_len, z) -> (m, n, z)
    expsum = rolling_matrix @ expvals
    # Return (n, m*z), flatten multiple computations to 2D
    df = pd.DataFrame(np.swapaxes(expsum, 0, 1).reshape((n, m*z)), index=data.index)
    df.columns = ['ems{}_'.format(zerolife) + col  for zerolife in zerolifes for col in data]
    return df


def apply_presum_transforms(df):
    """
    TRANSFORMATIONS
    - hematocrit * 100 for values < 1, do this first
    - eliminate outliers, do this first
    - take log of creatinine, do this first
    """
    try:
        df.loc[('R47841' ,   145398684 , 164), 'bp_sys_vitals'] = np.nan
        df.loc[('D1957987',  146011868,  101), 'maintenance_iv_volume_io'] = np.nan
        df.loc[('D1799062',  130290885 , 19), 'piggyback_iv_volume_io'] = np.nan

        df.loc[lambda df: df['hematocrit_labs'] < 1, 'hematocrit_labs'] = (
            df.loc[lambda df: df['hematocrit_labs'] < 1, 'hematocrit_labs']
            * 100)

        df['creatinine_labs'] = np.log(df['creatinine_labs'])
        df['hsa_enc'] = np.sqrt(df['hsa_enc'])
    except:
        print('\nFAILED TO APPLY PRESUM TRANSFORMS!\n')
        return df
    return df


def apply_postsum_transforms(df):
    """
    TRANSFORMATIONS
    - take log+1 of IO's
    - take log+1 of ems
    - take log+1 of tsl
    """
    pref_dict = generate_prefix_dict(df)
    suff_dict = generate_suffix_dict(df)

    log_plus_one = lambda s: np.log(s+1)
    curr_ios = list(set(pref_dict['curr']) & set(suff_dict['io']))
    df.loc[:, curr_ios] = df[curr_ios].apply(log_plus_one)
    df.loc[:, pref_dict['ems']] = df[pref_dict['ems']].apply(log_plus_one)
    df.loc[:, pref_dict['tsl']] = df[pref_dict['tsl']].apply(log_plus_one)
    return df


def summary_stats(df):

    col_dict = generate_suffix_dict(df)

    max_hour = df.reset_index().hour_since_adm.max()

    df_copy = df.copy()
    curr = df_copy[col_dict['vitals'] + col_dict['labs'] +
                   col_dict['io'] + col_dict['occ']].add_prefix('curr_')
    enc = df_copy[col_dict['enc']]

    df = df.reset_index(level=[0,1], drop=True).reindex(np.arange(max_hour + 1))

    # --- Optimized tsl ---
    io_df  = df[col_dict['io'] + col_dict['occ']]
    num_df = df[col_dict['vitals'] + col_dict['labs']]
    io_nan_mask = (io_df.fillna(0) == 0).astype(int).values
    num_nan_mask = num_df.isna().astype(int).values
    io_tsl_arr  = np.zeros((io_df.shape[0], io_df.shape[1]))
    num_tsl_arr = np.zeros((num_df.shape[0], num_df.shape[1]))
    for i in range(io_df.shape[0]):
        io_tsl_arr[i,:]  = (1 + io_tsl_arr[i-1,:])*io_nan_mask[i,:]
        num_tsl_arr[i,:] = (1 + num_tsl_arr[i-1,:])*num_nan_mask[i,:]

    io_df.loc[:,:] = io_tsl_arr
    num_df.loc[:,:] = num_tsl_arr
    # ---

    ems = pd.concat(
        (expsum(df[col_dict['io'] + col_dict['occ']].fillna(0), zerolifes = [z])
            for z in [6,24,72]),
        axis=1)

    # Take diff between curr and ema
    ema_vitals = pd.concat(
        (df[col_dict['vitals']].values -
         df[col_dict['vitals']].ewm(halflife=halflife).mean().add_prefix(
             'ema{}_'.format(halflife))
            for halflife in [6, 24, 72]),
        axis=1)

    ema_labs = pd.concat(
        (df[col_dict['labs']].values -
            df[col_dict['labs']].ewm(halflife=halflife).mean().add_prefix(
                'ema{}_'.format(halflife))
            for halflife in [12, 48, 144]),
        axis=1)


    ffill_img = pd.concat(
         (df[col_dict['img']].fillna(method='ffill', limit=lim).add_prefix(
             'ind{}_'.format(lim))
             for lim in [24,504]),
         axis=1).fillna(0)

    merged_df = pd.merge(
        pd.concat([enc, curr],axis=1),
        #curr,
        pd.concat([ema_vitals, ema_labs, ems,
                   num_df.add_prefix('tsl_'),
                   io_df.add_prefix('tsl_'),
                   ffill_img
                   ], axis=1),
        left_index=True, right_index=True, how='left')
    return merged_df

def ffill_curr(df):
    df = df.copy()
    pref_dict = generate_prefix_dict(df)
    suff_dict = generate_suffix_dict(df)
    df.loc[:,pref_dict['curr']] = (
        df[pref_dict['curr']]
        .fillna(method='ffill'))
    df.loc[:,suff_dict['img']] = (
        df[suff_dict['img']]
        .fillna(method='ffill', limit=6))

    return df

def fill_na(df, train_means):
    """
    PROTOCOL
    Curr - ffill then fill with training set means
    EMA - mean (should only be at beginning of encounter)
    EMS - 0
    """
    df = df.copy()
    pref_dict = generate_prefix_dict(df)
    df.loc[:,pref_dict['curr']] = (
        df[pref_dict['curr']]
        .fillna(train_means.loc[pref_dict['curr']]))
    df.loc[:,pref_dict['ema']] = (
        df[pref_dict['ema']]
        .fillna(train_means.loc[pref_dict['ema']]))

    # df['word_log_ratio_img'] = df['word_log_ratio_img'].fillna(
    #     train_means['word_log_ratio_img'])
    df.loc[:,pref_dict['img']] = (
         df[pref_dict['img']].fillna(0))
    return df





def scale(train, val, test):
    """
    SCALING PROTOCOL

    All TSL are MinMaxScaled

    EMS - StandardScaled, not LogNormal because outlier events are truly outliers
    EMA - StandardScaled
    TSL - LogScaled -> StandardScaled or MinMax


    """
    train, val, test = train.copy(), val.copy(), test.copy()
    pref_dict = generate_prefix_dict(train)
    suff_dict = generate_suffix_dict(train)

    curr_ios  = list(set(pref_dict['curr']) &
                    (set(suff_dict['io']) | set(suff_dict['occ'])))
    curr_nums = list(set(pref_dict['curr']) &
                    (set(suff_dict['vitals']) | set(suff_dict['labs'])))
    stand_cols  = pref_dict['ems'] + pref_dict['ema'] + ['age_enc'] + curr_nums
    minmax_cols = pref_dict['tsl'] + curr_ios + ['time_of_day_enc']

    scaler = StandardScaler()
    train.loc[:,stand_cols] = scaler.fit_transform(train.loc[:,stand_cols])
    val.loc[:,stand_cols]   = scaler.fit_transform(val.loc[:,stand_cols])
    test.loc[:,stand_cols]  = scaler.fit_transform(test.loc[:,stand_cols])

    minmax = MinMaxScaler()
    train.loc[:,minmax_cols] = minmax.fit_transform(train.loc[:,minmax_cols])
    val.loc[:,minmax_cols]   = minmax.fit_transform(val.loc[:,minmax_cols])
    test.loc[:,minmax_cols]  = minmax.fit_transform(test.loc[:,minmax_cols])

    return train.round(5), val.round(5), test.round(5)

def pipe_print(df,stage):
    print('\n')
    print('>>> PIPE PRINT: {}'.format(stage))
    print('Shape: ' + str(df.shape))
    print('Number of Patients: ' + str(df.reset_index().mrn.nunique()))
    print('Number of Encounters: '+ str(df.reset_index().id.nunique()))
    return df


def preprocess_exp_weights(
        rebuild=False,
        testing=False,
        time_to_event=True,
        time_varying=False,
        sbo500k=False,
        scale_feat=False,
        fill_null=False,
        ffill=True,
        custom_tag=None):
    '''
    rebuild -- Rebuild the dataframe and save it
    testing -- Use less data, less columns, and don't overwrite saved files
    time_to_event -- Returns dataset with surgery indicator, as well as
                     surg_datetime_hour_enc indicator
    time_varying -- Returns dataset with surgery indicator only for time of surgery
    '''
    flags = ''
    flags += '_mini' if testing else ''
    flags += '_500k' if sbo500k else ''
    flags += '_nan' if not fill_null else ''
    flags += '_ffill' if ffill else ''
    flags += '_unscaled' if not scale_feat else ''
    flags += ('_' + custom_tag) if custom_tag else ''
    print('Flags suffix: {}'.format(flags))

    start = time()

    if rebuild:
        print('\n>>> Rebuilding file...')

        if sbo500k:
            sbo = pd.read_pickle(proc_path + 'sbo500k.pickle')
        elif testing:
            sbo = (pd.read_pickle(proc_path + 'sbo_mini.pickle')
                   [['pulse_vitals', 'sodium_labs',
                     'tube_output_io', 'stool_occ']])
        else:
            sbo = (pd.read_pickle(proc_path + 'sbo.pickle')
                   .drop(['drain_output_io', 'ip_blood_administration_volume_io',
                      'maintenance_iv_bolus_volume_io', 'rectal_tube_output_io'], 1))

        # Initial processing
        sbo = (sbo
            .pipe(pipe_print, 'Merge Encounters')
            .pipe(merge_encounters)
            .pipe(pipe_print, 'Merge Imaging')
            .pipe(merge_imaging)
            .pipe(pipe_print, 'Standardize Times')
            .pipe(standardize_times)
            .pipe(pipe_print, 'Summary Stats')
            .pipe(fix_occurrences))

        # Write out file before doing summary stats
        sbo.to_pickle('data/processed/sbo_exp_presumm{}.pickle'.format(flags))

        # log transforms etc.
        sbo = apply_presum_transforms(sbo)

        # Main processing
        sbo = (
            sbo
            .groupby(level=[0,1])
            .apply(lambda df: summary_stats(df)))

        # Log transforms etc.
        sbo = apply_postsum_transforms(sbo)

        if fill_null or ffill:
            sbo = sbo.groupby(level=[0,1]).apply(ffill_curr)

        if fill_null:
            means = sbo.mean(axis=0)
            sbo = sbo.fillna(value=means.to_dict())

        # Write out preprocessed dataframe for visualization
        sbo.to_pickle('data/processed/sbo_exp_weights{}.pickle'.format(flags))
    else:
        print('\n>>> Reading in file...')
        sbo = pd.read_pickle('data/processed/sbo_exp_weights{}.pickle'.format(flags))

    if time_to_event:
        cols_to_drop = ['any_sbo_surg_enc', 'time_to_event_enc']
        surg_label = ['time_to_event_enc', 'any_sbo_surg_enc']
    else:
        cols_to_drop = ['any_sbo_surg_enc', 'time_to_event_enc']
        surg_label = ['any_sbo_surg_enc']

    sbo_x = sbo.drop(cols_to_drop, 1)
    sbo_y = sbo[surg_label]
    x_cols = list(sbo.drop(cols_to_drop,1).columns)
    print('\n\nFinished processing in ' + str(round(time()-start)))
    return sbo_x, sbo_y, x_cols

if __name__ == '__main__':
    sbo_x, sbo_y, x_cols = preprocess_exp_weights(
        rebuild=True, testing=False, scale_feat=False,
        fill_null=False, ffill=True, custom_tag='hand'
    )
    #sbo_test = preprocess(rebuild=False, testing=True)


