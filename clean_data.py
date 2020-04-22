"""
Clean raw data and merge.
"""

import pandas as pd
from time import time
import re
from clean_data import (clean_outcomes, clean_vitals, clean_labs,
                        clean_io, clean_io_unmeasured, build_encounters_df)

raw_path = 'data/raw/'
proc_path = 'data/processed/'

# Call cleaning functions
outcomes = clean_outcomes(raw_path + 'outcomes.csv')
outcomes.to_pickle(proc_path + 'outcomes.pickle')

encounters = build_encounters_df(outcomes)
encounters.to_pickle(proc_path + 'encounters.pickle')

vitals = clean_vitals([raw_path + 'vitals{}.csv'.format(i) for i in range(1,8)])
vitals.to_pickle(proc_path + 'vitals.pickle')

labs = clean_labs([raw_path + lab for lab in ['cbc.csv', 'cmp.csv', 'bmp.csv']])
labs.to_pickle(proc_path + 'labs.pickle')

io = clean_io([raw_path + io for io in ['io1.csv', 'io2.csv']])
io.to_pickle(proc_path + 'io.pickle')

io_unmeasured = clean_io_unmeasured(raw_path + 'io_unmeasured.csv')
io_unmeasured.to_pickle(proc_path + 'io_unmeasured.pickle')

# Merge
start = time()
sbo = (pd.concat([vitals, labs, io, io_unmeasured], axis=1)
   .sort_index(level=[0,2])
   .rename(columns = lambda x: re.sub(' ', '_', x)))
print('Merged dataframes in {}s'.format(round(time()-start)))

start = time()
sbo.to_pickle(proc_path + 'sbo.pickle')
print('Exported files in {}s'.format(round(time()-start)))




