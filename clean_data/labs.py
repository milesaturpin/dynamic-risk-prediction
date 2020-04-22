'''
Clean and return labs data.
'''

import pandas as pd
from time import time

def clean_labs(filenames):
	'''
	Clean and merge labs dfs...
	files -- list of filenames
	'''
	start = time()
	labs = (pd.concat(read_labs(f) for f in filenames)
			.pivot_table(index=['mrn', 'id','datetime'],
						 columns='component_name',
						 values='result_value')
			[components]
			.add_suffix('_labs'))
	print('Cleaned labs in {}s'.format(round(time()-start)))
	return labs

def read_labs(filename):
	'''
	Before merging...
	E.g. bmp.csv -> bmp dataframe
	'''
	return (pd.read_csv(filename, parse_dates=['RESULT_TIME'])
		   .rename(columns=str.lower)
		   #.drop('result_value', 1)
		   .rename(columns={'patient_mrn': 'mrn',
						   'pat_enc_csn_id': 'id',
						   'result_time': 'datetime',
						   #'result_num_value': 'result_value'
						  })
		   [['mrn', 'id', 'datetime', 'component_name', 'result_value']]
		   .assign(component_name = lambda x:
				   x['component_name']
				   .str.replace(' *\(.*\)', '')
				   .str.lower())
		   .assign(result_value = lambda df: df['result_value'].astype(str))
		   .assign(result_value = lambda df: df['result_value'].str.replace('\<|\>|=',''))
		   [lambda df: df['result_value'].str.contains('[A-Z]|[a-z]|\>|\<|\*|\+|\?|\%|\&|\(|\)') == False]
		   .assign(result_value = lambda df: df['result_value'].astype(float)))


components = ['platelet count', 'white blood cell count', 'hemoglobin', 'hematocrit', ### CBC
			  'sodium', 'potassium', 'chloride', ### CMP/BMP primary
			  'albumin', #'alkaline ahosphatase', 'anion gap',
			  'calcium', 'carbon dioxide',
			  'creatinine', #'remisol bun/creat ratio'] ### CMP/BMP secondary
			  ]
