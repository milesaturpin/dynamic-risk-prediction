"""
- Cut out non SBO surgery information
- Get time to surgery
- Categorize admitting divisions
"""
import pandas as pd
import numpy as np
from time import time
from utils import earliest_surg, unique_surg

def build_encounters_df(outcomes):
    """Build encounters dataframe."""


    start = time()
    # First SBO related surgery
    group1 = (outcomes
              [outcomes.cpt_code.isin(["49320", "49000", "49321", "49002"])]
              .pipe(unique_surg)
              .pipe(earliest_surg))

    # Earliest non SBO related surgeries and non surgery encounters
    group2 = (outcomes[~outcomes.any_sbo_surg]
              .pipe(unique_surg)
              .pipe(earliest_surg))

    # Get rid of surgery information
    group2[['surg_datetime', 'prim_proc', 'prim_proc_name',
            'prim_phys_name', 'surg_clin_div', 'surg_prov_name', 'serv_name']] = np.nan

    # Time to surgery
    encounters = (pd.concat([group1,group2])
        .sort_values(['adm_datetime'])
        .assign(surg_datetime   = lambda df: pd.to_datetime(df.surg_datetime))
        .assign(time_to_surg_td = lambda df: (df.surg_datetime - df.adm_datetime))
        .assign(time_to_surg    = lambda df: df.time_to_surg_td.apply(lambda x: x.total_seconds()/(24*3600)))
        .drop(['sbo_surg', 'non_sbo_surg'], 1))


    # Categorize admitting divisions
    adm_clin_div = pd.read_csv('~/projects/Duke Surgery - SBO/adm_clin_div.csv', index_col='adm_clin_div')

    divs = [list(adm_clin_div[adm_clin_div['cat'] == i].index) for i in [1,0]]

    def assign_cat(div):
        if div in divs[0]:
            return 'Surgical'
        elif div in divs[1]:
            return 'Medical'

    encounters['adm_clin_div_cat'] = encounters.adm_clin_div.apply(assign_cat)



    flow = pd.read_pickle('data/raw/flowsheet.pickle')

    def adm_time(row):
        if row['emer_datetime'] == row['emer_datetime']:
            return row['emer_datetime']
        elif row['hosp_adm_datetime'] == row['hosp_adm_datetime']:
            return row['hosp_adm_datetime']
        else:
            return row['adm_datetime']

    to_hour = lambda td: td.total_seconds()/3600
    to_day = lambda td: td.total_seconds()/(3600*24)

    print('updated eff_los')
    def eff_los(row):
        if row['time_to_surg'] != row['time_to_surg']:
            return to_day(row['disch_datetime'] - row['adm_datetime'])
            #return row['los']
        else:
            return to_day(row['surg_datetime'] - row['adm_datetime'])

    cols = {
        'patient_mrn'  : 'mrn',
        'prim_enc_csn_id'  : 'id',
        'emer_class_admit_time' : 'emer_datetime',
        'hosp_admsn_time'  : 'hosp_adm_datetime'
    }

    flow = (flow.rename(columns=str.lower)
                .rename(columns=cols)
                [list(cols.values())]
                 .drop_duplicates()
                 .set_index(['mrn', 'id']))

    encounters = (encounters.set_index(['mrn', 'id'])
          #[lambda df: df.sbo_poa]
          .merge(flow, left_index=True, right_index=True, how='left')
          .assign(adm_diff = lambda df: (df['adm_datetime'] - df.apply(adm_time, 1))
                  .apply(to_day))
          .assign(adm_datetime = lambda df: df.apply(adm_time, 1))
          .assign(los = lambda df: (df.disch_datetime - df.adm_datetime)
                                    .apply(to_day))
          .assign(eff_los = lambda df: df.apply(eff_los, 1)))


    print('Built encounters in {}s'.format(round(time()-start)))
    return encounters

