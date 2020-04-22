
import pandas as pd
from time import time

cols = {
	'patient_mrn'             : 'mrn',
	'age_at_dc'               : 'age',
	#'hsp_account_id'         : 'id',
	'prim_enc_csn_id'         : 'id',
	'tot_chgs'                : 'charges',
	'admission_datetime'      : 'adm_datetime',
	'adm_prov_name'           : 'adm_prov_name',
	'admitting_clin_div'      : 'adm_clin_div',
	'discharge_datetime'      : 'disch_datetime',
	'los'                     : 'los',
	'location'                : 'location',
	'disch_dept_name'         : 'disch_dept_name',
	'att_prov_name'           : 'att_prov_name',
	'att_prim_specialty'      : 'att_prim_spec',
	'att_clin_div'            : 'att_clin_div',
	'msdrg_name'              : 'drg',
	'sbo_dx_code'             : 'sbo_dx_code',
	'sbo_dx_nm'               : 'sbo_dx_name',
	'sbo_poa'                 : 'sbo_poa',
	'primary_procedure_id'    : 'prim_proc_id',
	'actual_start_time'       : 'surg_datetime',
	'primary_physician_nm'    : 'prim_phys_name', # surgeon
	'surg_clin_div'           : 'surg_clin_div',
	'service_name'            : 'serv_name',
	'cpt_code'                : 'cpt_code',
	'cpt_order_name'          : 'cpt_name',
	'prim_dx'                 : 'prim_dx',
	'prim_dx_name'            : 'prim_dx_name',
	'prim_proc'               : 'prim_proc', # need?
	'prim_proc_name'          : 'prim_proc_name', # need?
	'prim_proc_perf_prov_name': 'surg_prov_name',
	'adm_src'                 : 'adm_src',
	'adm_src_name'            : 'adm_src_name',
	'pat_stat'                : 'pat_stat',
	'pat_stat_name'           : 'pat_stat_name',
	'prim_payor_fin_class'    : 'prim_payor_fin_class'
}


'''
- Filter 18+
- Take out missing values
- Remove people who have 4+ encounters
- Add helpful columns
    - any_sbo_surg
    - sbo_surg
    - any_non_sbo_surg
    - non_sbo_surg
    - sbo_drg
    - ileus
'''

def clean_outcomes(f):
	start = time()

	outcomes = (pd.read_csv(f, parse_dates=['ADMISSION_DATETIME', 'DISCHARGE_DATETIME', 'ACTUAL_START_TIME'])
	            .drop('Unnamed: 0', 1)
	            .rename(columns=str.lower)
	            .rename(columns=cols)
	            [list(cols.values())]
	            # Filter 18+
	            .query('age >= 18')
	            .sort_values(['adm_datetime','surg_datetime'])
	            .assign(sbo_poa = lambda df: (df.groupby('id')
	                                          .sbo_poa
	                                          .transform(lambda x: (x == 'SBO on Admission'	).any())),
						adm_clin_div   = lambda df: df.adm_clin_div.str.title(),
						adm_prov_name  = lambda df: df.adm_prov_name.str.title(),
						att_clin_div   = lambda df: df.att_clin_div.str.title(),
						att_prov_name  = lambda df: df.att_prov_name.str.title(),
						surg_clin_div  = lambda df: df.surg_clin_div.str.title(),
						surg_prov_name = lambda df: df.surg_prov_name.str.title(),
						prim_phys_name = lambda df: df.prim_phys_name.str.title(),
						location       = lambda df: df.location.str.title(),)
	            .drop_duplicates()
				[['mrn', 'id', 'adm_datetime', 'surg_datetime', 'disch_datetime', 'age', 'los', 'sbo_poa', 'adm_clin_div',
				  'adm_prov_name',
				  'att_clin_div', 'att_prov_name', 'att_prim_spec', 'surg_clin_div', 'surg_prov_name', 'prim_phys_name',
				  'serv_name', 'sbo_dx_code',
				  'sbo_dx_name', 'drg', 'prim_dx', 'prim_dx_name', 'prim_proc_id', 'prim_proc', 'prim_proc_name', 'cpt_code',
				  'cpt_name',
				  'disch_dept_name', 'location', 'pat_stat', 'pat_stat_name', 'adm_src', 'adm_src_name', 'charges',
	              'prim_payor_fin_class']])

	# Take out missing values
	outcomes.loc[outcomes.mrn == 'FK7065', 'drg'] = 'MISSING'
	outcomes = outcomes.drop(outcomes[(outcomes['mrn'] == 'AH5676') & (outcomes['id'] == 163118220) &
	                                  (outcomes['surg_datetime'].isna())].index)
	outcomes = outcomes.drop(outcomes[(outcomes['mrn'] == 'FA3801') & (outcomes['id'] == 77222367) &
	                                  (outcomes['surg_datetime'].isna())].index)
	outcomes = outcomes.drop(outcomes[(outcomes['mrn'] == 'FF9296') & (outcomes['id'] == 150185867) &
	                                  (outcomes['surg_datetime'].isna())].index)
	outcomes = outcomes.drop(outcomes[(outcomes['mrn'] == 'RJ2679') & (outcomes['id'] == 74537839) &
	                                  (outcomes['surg_datetime'].isna())].index)

	# Remove people who have 4+ encounters
	unique_enc = outcomes[['mrn','id']].drop_duplicates().mrn.value_counts()
	outcomes = outcomes[outcomes.mrn.isin(unique_enc[unique_enc < 4].index)]

	def na_surg_placeholder(df):
		'''Fill in empty surg datetimes for intermediate calculation'''
		return (df.assign(surg_datetime = lambda df: (df.surg_datetime
		                                              .fillna(pd.to_datetime('1/1/1'))))
				.groupby(['id','surg_datetime'])
				.cpt_code)

	sbo_cpt = ["49320", "49000", "49321", "49002"]

	# Add helpful columns
	outcomes['any_sbo_surg'] = (outcomes.groupby('id').cpt_code
	                            .transform(lambda x: x.isin(sbo_cpt).any()))
	outcomes['sbo_surg'] = (na_surg_placeholder(outcomes)
	                        .transform(lambda x: x.isin(sbo_cpt).any()))
	outcomes['non_sbo_surg'] = (na_surg_placeholder(outcomes)
	                            .transform(lambda x: ((~x.isin(sbo_cpt))
	                                                  & (x.notna())).all()))
	outcomes['any_non_sbo_surg'] = (outcomes.groupby('id').non_sbo_surg
	                                .transform(lambda x: x.any()))
	outcomes['sbo_drg'] = outcomes.drg.str.contains('BOWEL|OBSTRUCTION')
	outcomes['ileus'] = (outcomes.groupby('id').sbo_dx_name
	                     .transform(lambda x: x.str.contains('Ileus|ileus').any()))
	print('Cleaned outcomes in {}s'.format(round(time()-start)))

	return outcomes




