"""
Scale data, split into X and y, return as numpy arrays.
"""

from utils import generate_col_dict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess(df, enc=False, drop_rates=False):

    col_dict = generate_col_dict(df)

    def cut_rows(df):
        los = df.shape[0]
        los = min(round(los/5), 12)
        return df.head(los)

    if enc:
        df = (df.reset_index()
              .groupby('id')
              .head(6))

        df = df.groupby('id').mean()
    print('test')
    print(df.shape)

    y = df['any_sbo_surg_enc']

    df = df.reset_index(drop=True)
    if drop_rates:
        sbo_numeric = df[col_dict['vitals'] + col_dict['labs']]
        sbo_binary  = df[col_dict['occ']]
        sbo_io      = df[col_dict['io']]
    else:
        sbo_numeric = df[col_dict['vitals'] + col_dict['vitals6mean'] +
                         col_dict['vitals12mean'] + col_dict['vitals24mean'] +
                         col_dict['labs'] + col_dict['labs6mean'] +
                         col_dict['labs12mean'] + col_dict['labs24mean']]
        sbo_binary = df[col_dict['occ'] + col_dict['occ6sum'] +
                        col_dict['occ12sum'] + col_dict['occ24sum']]
        sbo_io     = df[col_dict['io'] + col_dict['io6sum'] +
                        col_dict['io12sum'] + col_dict['io24sum']]

    X_numeric = sbo_numeric.values
    X_binary  = sbo_binary.values
    X_io      = sbo_io.values
    X_cols    = list(sbo_numeric.columns) + list(sbo_io.columns) + list(sbo_binary.columns)

    # Feature scaling
    standard_scaler = StandardScaler()
    X_numeric_scaled = standard_scaler.fit_transform(X_numeric)
    minmax_scaler = MinMaxScaler()
    X_io_scaled = minmax_scaler.fit_transform(X_io)
    X = np.concatenate([X_numeric_scaled, X_io_scaled, X_binary], 1)

    return csr_matrix(X), y, X_cols



