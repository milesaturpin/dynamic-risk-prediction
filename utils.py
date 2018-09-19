"""
Some helpful functions.
"""

import pandas as pd
import re
import json

def generate_col_dict(cols):
    """
    Returns a dictionary endings paired with column names
    with that ending. Used for indexing.

    e.g. ['apple_enc', '2_enc', 'toast_arg', 'ahh_arg'] ->
    {'arg': ['toast_arg', 'ahh_arg'], 'enc': ['apple_enc', '2_enc']}
    """
    ends = set([re.search('(?<=_)[a-z|0-9]*$', col)[0] for col in cols])
    return {end: [col for col in cols if re.search(f'_{end}$', col)] for end in ends}

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