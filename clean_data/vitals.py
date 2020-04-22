'''
Clean and return vitals data.
'''

import pandas as pd
from time import time

def clean_vitals(filenames):
	'''
	Clean and merge vitals dfs...
	filenames -- list of filenames
	'''

	start = time()
	vitals = (pd.concat((read_vitals(f) for f in filenames))
			  .assign(meas_value = lambda df: pd.to_numeric(df.meas_value))
			  .pivot_table(index=['mrn', 'id','datetime'],
			  			   columns='disp_name',
			  			   values='meas_value')
			  .groupby(level=[0,1,2])
			  .mean()
			  .add_suffix('_vitals'))

	print('Cleaned vitals in {}s'.format(round(time()-start)))
	return vitals


def read_vitals(filename):
	'''
	E.g. vitals1.csv -> vitals1 dataframe
	'''
	print(filename)
	df = (pd.read_csv(filename, parse_dates=['RECORDED_TIME'])
		  .rename(columns=str.lower)
		  .rename(columns={'patient_mrn'   : 'mrn',
				  		   'pat_enc_csn_id': 'id',
						   'recorded_time' : 'datetime'})
		  .drop(['unnamed: 0'], axis=1)
		  .assign(disp_name = lambda df: df.disp_name.str.lower()))

	if 'vitals2' in filename:
		df = melt_bp(df)

	return df.set_index(['mrn', 'id', 'datetime'])

def melt_bp(df):
	return (df.drop(['disp_name', 'meas_value'], 1)
			.rename(columns={'systolic': 'bp_sys',
							 'diastolic': 'bp_dia'})
			.melt(id_vars = ['mrn', 'id', 'datetime'],
				  var_name='disp_name',
				  value_name='meas_value'))