"""
Split dataset on patient level.
"""

import numpy as np
import pandas as pd
from utils import to_json
from sklearn.model_selection import train_test_split
import argparse
from scipy.stats import bernoulli

def resolve(pat):
    """
    - Input patient with multiple encounters
    - Get outcome of encounters
    - Assign label of 1 with some probability according to proportion of
    encounters that ended in surgery
    """
    prob = (pat.any_sbo_surg_enc
            .value_counts()
            .transform(lambda x: x/x.sum())
            .reindex([True, False])
            .fillna(0)
            .loc[True])

    label = bernoulli.rvs(prob)
    return pd.Series([pat.iloc[0,0], label], index=['mrn', 'any_sbo_surg_enc'])

def train_val_test(sbo, seed):
    print(sbo)
    print("Number of encounters:" + str(sbo.reset_index().mrn.nunique()))
    print("Number of patients:" + str(sbo.reset_index().id.nunique()))

    # TODO: SORT THIS SHIT OUT
    np.random.seed(seed)
    pats = (sbo.reset_index()
            [['mrn', 'any_sbo_surg_enc']]
            .groupby('mrn')
            .apply(resolve))
    print('Full dataset proportion')
    print(sbo.any_sbo_surg_enc
            .value_counts()
            .transform(lambda x: x/x.sum()))
    print('Patient dataset proportion')
    print(pats.any_sbo_surg_enc
            .value_counts()
            .transform(lambda x: x/x.sum()))

    train_val_idx, test_idx, train_val_surg, _ = train_test_split(pats.mrn, pats.any_sbo_surg_enc,
                                                                              stratify=pats.any_sbo_surg_enc,
                                                                              test_size=0.1, random_state=seed)

    train_idx, val_idx, _, _ = train_test_split(train_val_idx, train_val_surg,
                                                            stratify=train_val_surg,
                                                            test_size=0.1/0.9, random_state=(seed+1))

    '''
    mrns = sbo.reset_index().mrn.drop_duplicates()
    train_val_idx, test_idx, train_val_surg, _ = train_test_split(mrns, mrns,
                                                                              #stratify=pats.any_sbo_surg_enc,
                                                                              test_size=0.1, random_state=seed)

                train_idx, val_idx, _, _ = train_test_split(train_val_idx, train_val_surg,
                                                            #stratify=train_val_surg,
                                                            test_size=0.1/0.9, random_state=(seed+1))
    '''

    train = sbo.loc[train_idx.values, :]
    val   = sbo.loc[val_idx.values, :]
    test  = sbo.loc[test_idx.values, :]

    for df in [train,val,test]:

        print(df.shape[0]/sbo.shape[0])
        print(df.any_sbo_surg_enc
              .value_counts().transform(lambda x: x/x.sum()))

    idx_dict = {'train': list(train_idx.values),
                'val': list(val_idx.values),
                'test': list(test_idx.values)}
    print('COMMENTED OUT')
    #to_json(idx_dict, 'references/idx_dict.json')

    return train, val,  idx_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, help='Add a random seed.')
    args = parser.parse_args()

    encounters = (pd.read_pickle('data/processed/encounters.pickle')
                  [lambda df: (df.sbo_poa)]
                  .set_index('mrn')
                  .rename(columns={'any_sbo_surg':'any_sbo_surg_enc'})
                  [lambda df: (df.time_to_surg > 2) |
                  ((df.time_to_surg != df.time_to_surg)&(df.los>2))]
                  )

    train, val, test, idx_dict = train_val_test (encounters)






