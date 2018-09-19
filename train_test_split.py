"""
Split dataset on patient level.
"""

import numpy as np
from utils import to_json
from sklearn.model_selection import train_test_split

np.random.seed(9172018)

def train_val_test(sbo):

	pats = (sbo.reset_index()
	        [['mrn', 'any_sbo_surg_enc']]
	        .fillna(0)
	        .drop_duplicates())

	train_val_idx, test_idx, train_val_surg, _ = train_test_split(pats.mrn, pats.any_sbo_surg_enc, stratify=pats.any_sbo_surg_enc, test_size=0.1, random_state=9172018)

	train_idx, val_idx, _, _ = train_test_split(train_val_idx, train_val_surg, stratify=train_val_surg, test_size=0.1/0.9, random_state=91720181)

	print(train_idx.shape)

	'''
	TODO: Check for covariate shift using classifier
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
	to_json(idx_dict, 'references/idx_dict.json')

	return train, val, idx_dict