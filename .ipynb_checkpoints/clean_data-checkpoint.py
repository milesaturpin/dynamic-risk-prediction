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
encounters = build_encounters_df(outcomes)
vitals   = clean_vitals([raw_path + f'vitals{i}.csv'for i in range(1,8)])
labs     = clean_labs([raw_path + lab for lab in ['cbc.csv', 'cmp.csv', 'bmp.csv']])
io       = clean_io([raw_path + io for io in ['io1.csv', 'io2.csv']])
io_unmeasured = clean_io_unmeasured(raw_path + 'io_unmeasured.csv')


# Export individual files
start = time()
outcomes.to_pickle(proc_path + 'outcomes.pickle')
encounters.to_pickle(proc_path + 'encounters.pickle')
vitals.to_pickle(proc_path + 'vitals.pickle')
labs.to_pickle(proc_path + 'labs.pickle')
io.to_pickle(proc_path + 'io.pickle')
io_unmeasured.to_pickle(proc_path + 'io_unmeasured.pickle')
print(f'Exported files in {round(time()-start)}s')

# Merge
start = time()
sbo = (pd.concat([vitals, labs, io, io_unmeasured], axis=1)
	   .sort_index(level=[0,2])
	   .rename(columns = lambda x: re.sub(' ', '_', x)))
print(f'Merged dataframes in {round(time()-start)}s')

start = time()
sbo.to_pickle(proc_path + 'sbo.pickle')
print(f'Exported files in {round(time()-start)}s')




