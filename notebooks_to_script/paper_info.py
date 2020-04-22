#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


outcomes = pd.read_pickle('data/processed/outcomes.pickle')


# In[15]:


outcomes['adm_datetime'].apply(lambda dt: dt.year).value_counts()


# In[2]:


pd.set_option('max_columns', 50)


# In[ ]:


sbo = pd.read_pickle('data/processed/sbo_exp_weights_nan_ffill_unscaled_noimg.pickle')


# In[19]:


sbo = pd.read_pickle('data/processed/sbo_exp_weights_nan_ffill_unscaled_hand.pickle')


# In[17]:


sbo = pd.read_pickle('data/processed/sbo_exp_weights_nan_ffill_unscaled_pruned.pickle')


# In[8]:


list(sbo.columns)


# In[4]:


sbo_raw = pd.read_pickle('data/processed/sbo.pickle')


# In[317]:


sbo_raw = (sbo_raw.reset_index()[lambda df: df['id'].isin(
    sbo.groupby(level=[0,1]).head(1).reset_index(level=2, drop=True)
    .index.get_level_values(1))].set_index(['mrn','id','datetime']))


# In[318]:


sbo_raw.reset_index().id.nunique()


# In[5]:


from utils import generate_prefix_dict, generate_midfix_dict, generate_suffix_dict


# In[28]:


pd.__path__


# In[26]:


sbo.reset_index().mrn.nunique()


# In[27]:


sbo.reset_index().id.nunique()


# In[23]:


sbo_raw.reset_index().mrn.nunique()


# In[25]:


sbo_raw.reset_index().id.nunique()


# In[29]:


import re


# In[33]:


re.search('r.d', 'red').group()


# ### outcomes

# In[6]:


enc=pd.read_pickle('data/processed/encounters.pickle')


# In[131]:


(enc[enc.sbo_poa].reset_index().mrn.value_counts() > 1).sum()


# In[132]:


enc[enc.sbo_poa].reset_index().mrn.nunique()


# In[133]:


454/3405


# # VARIABLE missingness

# In[244]:


sbo_presum = pd.read_pickle('data/processed/sbo_exp_presumm_nan_ffill_unscaled_noimg.pickle')
suff_dict = generate_suffix_dict(sbo_presum)


# In[45]:


sbo.index.get_level_values(1).unique()


# In[55]:


from utils import generate_prefix_dict


# In[56]:


pref_dict = generate_prefix_dict(sbo)


# In[18]:


sbo.shape


# In[286]:


temp = sbo_presum[suff_dict['vitals'] + suff_dict['labs']].groupby(level=1).mean().count().to_frame()
temp[1] = (temp[0] / temp[0].max()).round(3)
temp.apply(lambda row: '{:d} ({:.3f})'.format(int(row[0]),row[1]),1).to_frame()


# In[288]:


temp = (3910 - (sbo_presum[suff_dict['io'] + suff_dict['occ']].groupby(level=1).mean() == 0).sum()).to_frame()
temp[1] = (temp[0]/3910).round(3)
temp.apply(lambda row: '{:d} ({:.3f})'.format(int(row[0]),row[1]),1).to_frame()


# In[ ]:





# In[49]:


sbo_raw[sbo_raw.index.get_level_values(1).isin(sbo.index.get_level_values(1).unique())].shape


# ## VARIABLES MEANS

# In[287]:


(sbo_presum[suff_dict['vitals'] + suff_dict['labs']].describe().T.apply(
    lambda row: '{:1.2f}±{:1.2f} ({}-{})'.format(row['mean'], row['std'], row['min'], row['max']), 1)
 .str.replace('\.0', '').to_frame())


# In[285]:



(sbo_presum[suff_dict['io'] + suff_dict['occ']].apply(lambda s: s[s!=0]).describe()
 .T.apply(
    lambda row: '{:1.2f}±{:1.2f} ({}-{})'.format(row['mean'], row['std'], row['min'], row['max']), 1)
 .str.replace('\.0', '').to_frame())


# In[389]:


(sbo_raw.head(30000)['pulse_vitals'].dropna().reset_index(level=2)[['datetime']]
.groupby(level=[0,1]).diff().mean()[0].total_seconds() /3600)


# ### time bt measures

# In[ ]:





# In[381]:


get_ipython().run_cell_magic('time', '', "def time_bt_measures(df):\n    return (\n        df.apply(time_bt_measures_helper))\n\ndef time_bt_measures_helper(s):\n    s2 = s.dropna()\n    if s2 is not None and len(s2)>1:\n        return (s2.reset_index(level=2)\n            ['datetime']\n            .diff().reset_index(level=[0,1])\n               )\n    else:\n        s = s.head(1).reset_index(level=2)\n        s['datetime'] = np.nan\n        return s['datetime']\n\n\ndisplay(sbo_raw.head(3000)[\n    suff_dict['vitals'] + suff_dict['labs']].apply(\n    lambda s: s.to_frame().groupby(level=[0,1]).apply(time_bt_measures_helper)\n        .mean()).round(2).to_frame())")


# In[341]:


sbo_raw.shape


# In[330]:


_[0]


# In[290]:


sbo_presum.columns


# In[296]:


sbo_raw


# In[292]:


results = []
for col in sbo_presum[suff_dict['vitals'] + suff_dict['labs']]:
    mean_time = (sbo_presum[col].dropna().reset_index(level=2)[['hour_since_adm']]
 .groupby(level=1).apply(lambda df: df.diff().mean())).mean()[0]
    results.append([col, round(mean_time, 2)])
results=np.array(results)
pd.Series(results[:,1], index=results[:,0]).to_frame()


# In[112]:


from utils import generate_suffix_dict, generate_midfix_dict


# In[114]:


suff_dict = generate_suffix_dict(sbo)
mid_dict =  generate_midfix_dict(sbo)


# In[116]:


pref_dict.keys()


# In[149]:


sbo_raw.head(1000)[['bp_dia_vitals']].dropna().reset_index(level=2)[['datetime']].groupby(level=1).diff()


# In[152]:


to_hour = lambda td: td.total_seconds() / 3600


# In[161]:


for col in sbo_raw.columns:
    mean_time = (sbo_raw.head(1000)[col].dropna().reset_index(level=2)[['datetime']]
 .groupby(level=1).apply(lambda df: df.diff().dropna())['datetime'].apply(to_hour, axis=1)).mean()[0]
    print(col, mean_time)


# ### prune vocabs

# In[102]:


ct = pd.read_pickle('vocabularies/13gram_24surg_27nonsurg.pickle')
sboimg = pd.read_pickle('vocabularies/sboimg_23gram_min12_43surg_52nonsurg.pickle')


# In[103]:


ct


# In[106]:


ct_pruned = {'non_surg': ['ileus', 'is difficult', 'contrast axial', 'bladder wall',
        'early or partial', 'bladder wall thickening', 'malignant',
        'changed from', 'vertebral', 'malignant neoplasm of',
        'no ascites', 'fever', 'in size of', 'early or', 'vertebral body',
        'carcinomatosis', 'most consistent',
        'peritoneal carcinomatosis', 'nodularity', 'unchanged there',
        'small left pleural', 'impression no', 'metastases', 'necrosis',
        'acute pancreatitis'],
 'surg': ['closed loop obstruction',
        'closed', 'surgical consultation',
        'transition points', 'internal hernia', 'is transition point',
        'impression high grade', 'gastric bypass', 'impression high',
        'ischemia', 'two', 'with high', 'of the mesentery',
        'and contour no', 'pneumatosis or free',
         'quadrant with', 'loop of bowel',
        'bowel ischemia']}


# In[104]:


sboimg


# In[105]:


sboimg_pruned = {'non_surg': ['partial sbo', 'gaseous distention', 'gaseous distention of',
        'mid small', 'loop of', 'bowel transit time', 'mid small bowel',
        'small bowel transit', 'bowel transit', 'normal transit',
        'opacification of the', 'there is normal', 'mildly dilated loops',
        'mildly dilated', 'of the terminal', 'distention of',  'the ascending colon',
        'bowel gas pattern',  'ascending colon', 'colon at',
        'seconds timed',  'impression no',
        'impression delayed',  'normal appearance',
        'over the stomach', 'generalized abdominal pain', 'tube tip',
        'evidence of obstruction', 'to suggest', 'soluble contrast was', 'appearance of',
        'transit time was', 'with partial small'],
 'surg': ['slow transit', 'within the proximal',
        'history small bowel', 'findings preliminary',
        'bowel at', 'for small bowel', 'of the contrast',
        'not yet', 'upper abdomen', 'ct scan',
        'minutes contrast', 'dilution of',
        'obstruction recommend', 'obstruction comparison none',
        'small bowel contrast', 'technique single', 'the proximal jejunum',
        'technique single contrast', 'and proximal', 'obstruction evaluate',
        'serial radiographs', 'bowel contrast',
         'large bowel']}


# In[107]:


import pickle
with open('vocabularies/ct_pruned.pickle', 'wb') as f:
    pickle.dump(ct_pruned, f)

with open('vocabularies/sboimg_pruned.pickle', 'wb') as f:
    pickle.dump(sboimg_pruned, f)


# #### hand pick words

# In[3]:


ct_hand = {
    'non_surg': ['no ascites','ileus','early or partial', 'unchanged'],
    'surg': ['ischemia','closed loop obstruction','impression high grade','internal hernia',
          'transition point','pneumatosis or free']}


# In[4]:


sboimg_hand = {
    'non_surg': ['partial', 'bowel transit', 'normal transit','gaseous distention',
                 'opacification','normal appearance'],
     'surg': []}


# In[5]:


import pickle
with open('vocabularies/ct_hand.pickle', 'wb') as f:
    pickle.dump(ct_hand, f)

with open('vocabularies/sboimg_hand.pickle', 'wb') as f:
    pickle.dump(sboimg_hand, f)


# # Compare demographic distributions

# In[56]:





# In[222]:


demog = pd.read_excel('data/processed/SBO_PatientDemographics_091219.xlsx')

cols_to_drop = [
    'AGE_AT_ADMISSION',
    'PAT_NAME', 'PAT_ID', 'BIRTH_DATE',
    'HOSP_ADMSN_TIME', 'HOSP_DISCH_TIME','HOSP_STAY_DAYS'
]

col_map = {
    'MRN' : 'mrn',
    'PAT_ENC_CSN_ID' : 'id',
    'SEX' : 'sex',
    'ETHNICITY' : 'ethnicity',
    'RACE' : 'race' ,
    'HEIGHT' : 'height',
    'WEIGHT' : 'weight',
    'BMI' : 'bmi',
    'PATIENT_CLASS' : 'class',
}

ethn_map = {
              'Unavailable' : 'Not Reported/Declined',
         'Hispanic Mexican' : 'Hispanic or Latino',
    'Hispanic Puerto Rican' : 'Hispanic or Latino',
           'Hispanic Other' : 'Hispanic or Latino',
           'Hispanic Cuban' : 'Hispanic or Latino',
}

race_map = {
    'Unavailable' : 'Not Reported/Declined',
    'American Indian' : 'American Indian or Alaskan Native'
}

def map_variables(x, mapper):
    try:
        new = mapper[x]
    except:
        new = x
    return new

demog = (
    demog
    .drop(cols_to_drop, axis=1)
    .rename(columns=col_map)
    .dropna(subset=['mrn'])
    .set_index(['mrn', 'id']))

demog['ethnicity'] = demog['ethnicity'].apply(lambda x: map_variables(x, ethn_map), 1)
demog['race'] = demog['race'].apply(lambda x: map_variables(x, race_map), 1)

no_dup, dup = (
    demog.groupby(level=[1]).filter(lambda df: df.shape[0]==1),
    demog.groupby(level=[1]).filter(lambda df: df.shape[0]>1))

# some people have 2 listings of races, just pick one
race_reduced = dup.groupby(level=1).head(1)

demog = pd.concat([no_dup, race_reduced])


# In[224]:


demog.shape


# In[223]:


get_col_mapping = lambda s: pd.Series(s.unique(), index=s.cat.codes.unique())
get_col_mapping(demog['ethnicity'].astype('category'))


# In[125]:


# sex_mapping = get_col_mapping(demog['sex'].astype('category'))
# demog['sex'] = demog['sex'].astype('category').cat.codes

# eth_mapping = get_col_mapping(demog['ethnicity'].astype('category'))
# demog['ethnicity'] = demog['ethnicity'].astype('category').cat.codes

# race_mapping = get_col_mapping(demog['race'].astype('category'))
# demog['race'] = demog['race'].astype('category').cat.codes

# class_mapping = get_col_mapping(demog['class'].astype('category'))
# demog['class'] = demog['class'].astype('category').cat.codes


# In[126]:





# In[127]:





# In[128]:





# In[226]:


demog.head()


# In[227]:


sbo_id = sbo.reset_index()[['mrn', 'id']].drop_duplicates().values
sbo_id = zip(sbo_id[:,0], sbo_id[:,1])
enc_sbo = enc.loc[sbo_id]


# In[228]:


demog.reset_index().shape, demog.reset_index()[['mrn','id']].drop_duplicates().shape


# In[229]:


demog['race'].value_counts()


# In[230]:


enc_sbo.shape


# In[231]:


demog = pd.concat([demog, enc_sbo], axis=1, join='inner')


# In[232]:


demog.head()


# In[233]:


def strat_crosstab(col):
    def format_crosstab(row):
        nf, nt, pf, pt = row.values
        return pd.Series([
            '{} ({})'.format(int(nf), pf),
            '{} ({})'.format(int(nt), pt)],
            index=[False, True])

    crosstab = pd.crosstab(demog[col], demog['any_sbo_surg'])
    crosstab_norm = crosstab.transform( lambda x: x/x.sum()).round(3)
    display(pd.concat([crosstab, crosstab_norm], 1).apply(format_crosstab, axis=1))

strat_crosstab('race')


# In[234]:


from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(pd.crosstab(demog['race'], demog['any_sbo_surg']))
chi2, p


# In[235]:


strat_crosstab('ethnicity')


# In[236]:


from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(pd.crosstab(demog['ethnicity'], demog['any_sbo_surg']))
chi2, p


# In[237]:


strat_crosstab('sex')


# In[238]:


from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(pd.crosstab(demog['sex'], demog['any_sbo_surg']))
chi2, p


# In[239]:


strat_crosstab('location')


# In[199]:


from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(pd.crosstab(demog['location'], demog['any_sbo_surg']))
chi2, p


# In[220]:


strat_crosstab('prim_payor_fin_class')


# In[205]:


from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(pd.crosstab(demog['prim_payor_fin_class'], demog['any_sbo_surg']))
chi2, p


# In[ ]:





# In[167]:


sbo.head()


# In[168]:


from scipy.stats import mannwhitneyu


# In[202]:


(demog
 .groupby('any_sbo_surg')
 #[['age_enc', 'past_sbo_enc', 'duke_loc_enc', 'raleigh_loc_enc', 'regional_loc_enc']]
 [['age', 'los', 'weight']]
 .describe()
 .T)


# In[194]:


def mann_whitney(s, split_col='any_sbo_surg'):
    df_copy = s.reset_index(level=-1).copy()
    x = df_copy[df_copy[split_col] == 1].drop('any_sbo_surg', 1).values
    y = df_copy[df_copy[split_col] == 0].drop('any_sbo_surg', 1).values
    mwu = mannwhitneyu(x, y, alternative='two-sided')
    return pd.Series(mwu, index=['stat', 'pval'])


# In[203]:


(demog
 .set_index('any_sbo_surg')
 [['age', 'los', 'weight']]
 .apply(mann_whitney))


# In[198]:


1.426823e+06


# In[20]:


len(sbo.columns)


# # features

# In[189]:


#sbo.groupby('any_sbo_surg_enc').describe().stack()


# In[21]:


for col in sbo.columns:
    print(col)
