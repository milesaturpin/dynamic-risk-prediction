'''
Clean and return imaging data.
'''

import pandas as pd

def read_imaging(proc_path, filename):

	imaging = (pd.read_csv(filename)
				.rename(columns=str.lower)
				.rename(columns=cols)
				[list(cols.values())]
				.assign(datetime = lambda df: pd.to_datetime(df['datetime']))
				.pipe(join_narrative))

	imaging.to_pickle(proc_path + 'imaging.pickle')

	return imaging

cols = {'patient_mrn': 'mrn', 'pat_enc_csn_id': 'id',
		'result_time': 'datetime', 'lab_name': 'image_name',
		'line': 'line', 'narrative': 'narrative'}

def join_narrative(df):
	return (df.groupby(['mrn', 'id', 'datetime', 'image_name'])
			.narrative
			.apply(lambda x: '\\'.join(x).strip())
			.reset_index(level=3))

