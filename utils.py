"""
Some helpful functions.
"""

import pandas as pd
import re
import json

def generate_suffix_dict(df):
    """
    Returns a dictionary endings paired with column names
    with that suffix. Used for indexing.

    e.g. ['apple_enc', '2_enc', 'toast_arg', 'ahh_arg'] ->
    {'arg': ['toast_arg', 'ahh_arg'], 'enc': ['apple_enc', '2_enc']}
    """
    endings = set([re.search('(?<=_)[a-z|0-9]*$', col).group() for col in df])
    return {end: [col for col in df if re.search('_{}$'.format(end), col)] for end in endings}

def generate_prefix_dict(df):
    """
    Returns a dictionary endings paired with column names
    with that prefix. Used for indexing.

    e.g. ['apple_enc', 'apple_2', 'arg_toast', 'arg_ahh'] ->
    {'apple': ['apple_enc', 'apple_2'], 'arg': ['arg_toast', 'arg_ahh']}
    """
    starts = set([re.search('^[a-z]*(?=\d|_)', col).group() for col in df])
    return {start: [col for col in df if re.search('^{}(\d|_)'.format(start), col)] for start in starts}

def generate_midfix_dict(df):
    """
    Returns a dictionary endings paired with column names
    with that prefix. Used for indexing.

    e.g. ['apple_enc', 'apple_2', 'arg_toast', 'arg_ahh'] ->
    {'apple': ['apple_enc', 'apple_2'], 'arg': ['arg_toast', 'arg_ahh']}
    """
    #print(set([re.sub('^[a-z|0-9]*_', '', col) for col in df]))
    mids = set([re.sub('_[a-z|0-9]*$',
                        '',
                       re.sub('^[a-z|0-9]*_', '', col)) for col in df])
    return {mid: [col for col in df if re.search(mid, col)] for mid in mids}

def to_json(some_dict, file_path):
    """
    Convert dict to JSON and store in references directory.
    """
    with open(file_path, 'w') as out:
        json.dump(some_dict, out)


# Outcomes dataset
def most_recent_encounter(df):
    return (df.loc[df.groupby('mrn')
                 .adm_datetime
                 .idxmax()])

def sbo_related_surg(df):
    return df.query('cpt_code in ["49320", "49000", "49321", "49002"] | cpt_code != cpt_code')

def non_sbo_related_surg(df):
    return df.query('cpt_code not in ["49320", "49000", "49321", "49002"] | cpt_code != cpt_code')

def sbo_drg(df):
    return df[df.drg.str.contains('BOWEL|OBSTRUCTION')]

def earliest_surg(df):
    surg_mask = df['surg_datetime'].notna()
    surg = df[surg_mask]
    no_surg = df[~surg_mask]

    surg = surg.loc[surg.groupby('id')
                    .surg_datetime
                    .idxmin()
                    .dropna()
                    .astype(int)]
    return pd.concat([surg, no_surg])

def unique_surg(df):
    return (df.drop(['cpt_code','cpt_name',
                     'prim_proc_id',
                     'sbo_dx_code', 'sbo_dx_name'], 1)
            .drop_duplicates())

def get_duplicates(df, indx):
    return df.loc[df.set_index(indx)
                  .index
                  .duplicated(False)]
