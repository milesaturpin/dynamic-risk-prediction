'''
Clean and return io data.
'''

import pandas as pd
from time import time

def clean_io(filenames):
    start = time()
    #print('TODO: Drop columns')
    io = (pd.concat(read_io(f) for f in filenames)
            .assign(flo_meas_name = lambda x:
                       x['flo_meas_name']
                       .str.lower()
                        .str.replace('^r ', ''))
                .pivot_table(index=['mrn', 'id','datetime'],
                         columns='flo_meas_name',
                         values='meas_value')
            .drop([
                  'duhs lda tube residual discard', 'duhs lda tube flush',
                   'epidural intake',
                    # removed these recently
                   'duhs fistula output', 'chest tube output', 'drain intake',
                    'duhs lda rectal tube intake', 'ostomy output', 'urostomy output',

                   'chemo iv volume', 'duhs ip blood volume (manual entry)',
                'epidural bolus dose','ip mtp other blood products volume',
                   'ip mtp platelets volume','ip mtp cryoprecipitate volume',
                   'ip mtp prbc volume'
                  ], 1)
            .add_suffix('_io'))
    print('Cleaned io in {}s'.format(round(time()-start)))
    return io

def read_io(filename):
    return (pd.read_csv(filename, parse_dates=['RECORDED_TIME'])
            .rename(columns=str.lower)
               .rename(columns={'patient_mrn'   : 'mrn',
                            'pat_enc_csn_id': 'id',
                            'recorded_time'  : 'datetime'})
            [['mrn', 'id', 'datetime', 'flo_meas_name', 'meas_value']])
