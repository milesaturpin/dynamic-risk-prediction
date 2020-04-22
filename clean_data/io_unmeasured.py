'''
Clean and return io unmeasured data.
'''

import pandas as pd
import re
from time import time

def clean_io_unmeasured(f):
	start = time()
	io_unmeasured = (read_io_unmeasured(f)
					 .assign(disp_name = lambda x: x['disp_name'].str.lower())
					 .pivot_table(index=['mrn', 'id','datetime'],
					 			  columns='disp_name',
					 			  values='meas_value')
					 .rename(lambda x: re.sub('occurrence', 'occ', x), axis=1))
	print('Cleaned io_unmeasured in {}s'.format(round(time()-start)))
	return io_unmeasured

def read_io_unmeasured(filename):
	return (pd.read_csv(filename, parse_dates=['RECORDED_TIME'])
			.rename(columns=str.lower)
			.rename(columns={'patient_mrn' : 'mrn',
						   'pat_enc_csn_id': 'id',
						   'recorded_time' : 'datetime'})
			[['mrn', 'id', 'datetime', 'disp_name', 'meas_value']])