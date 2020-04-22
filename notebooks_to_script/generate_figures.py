#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glmnet_python
from glmnet import glmnet
from glmnet_python import glmnetPlot, glmnetPredict, glmnetPrint, glmnetCoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from IPython import InteractiveShell
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
#import seaborn as sns
import time
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import interp
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, precision_recall_curve

from utils import generate_prefix_dict, generate_suffix_dict, generate_midfix_dict

import lifelines

InteractiveShell.ast_node_interactivity = "all"


# In[2]:


plt.rcParams.update({'font.size': 14})


# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from preprocessing_exp_weights import preprocess_exp_weights

(x, y, x_cols) = preprocess_exp_weights(
    rebuild=False, time_to_event=True, scale_feat=False, fill_null=False,testing=False, custom_tag='hand')
#sboimg

suff_dict = generate_suffix_dict(x)
#x = x.drop(list(set(suff_dict['img'])-{'ind12_word_log_ratio_img','ind48_word_log_ratio_img'}), 1)

pref_dict = generate_prefix_dict(x)
suff_dict = generate_suffix_dict(x)
mid_dict = generate_midfix_dict(x)
#mid_dict['bp'] = mid_dict['bp_sys'] + mid_dict['bp_dia']


# In[4]:


def auc_over_time(y_pred, fold_dict,
                  tte_q=12, tte_range=24,
                  hsa_q=12, hsa_range=(0,240)):

    y = y_pred.copy()

    y['hsa'] = y.index.get_level_values(2)
    y['tte_qcut'] = np.nan
    y['hsa_qcut'] = np.nan

    y.loc[y.time_to_event_enc < tte_range, 'tte_qcut'] = (
        pd.qcut(y.loc[
            lambda df: df.time_to_event_enc < tte_range, 'time_to_event_enc'],
            q=tte_q)
        .apply(lambda x: x.right)
        .astype(int))

    y.loc[(hsa_range[0] <= y.hsa) & (y.hsa < hsa_range[1]), 'hsa_qcut'] = (
        pd.qcut(y.loc[
            lambda df: (hsa_range[0] <= df.hsa) & (df.hsa < hsa_range[1]), 'hsa'],
            q=hsa_q)
        .apply(lambda x: x.right)
        .astype(int))

    tte_bin_labels = y['tte_qcut'].dropna().sort_values().unique()
    hsa_bin_labels = y['hsa_qcut'].dropna().sort_values().unique()

    y['target']  = (
        y.any_sbo_surg_enc & (y.time_to_event_enc < tte_range)).astype(int)

    num_folds = len(fold_dict)
    tte_auc_arr = np.zeros((tte_q, num_folds))
    hsa_auc_arr = np.zeros((hsa_q, num_folds))

    for i in range(num_folds):
        print('Fold {}'.format(i))
        start = time.time()
        y_i = y[y.fold == i]
        #y_i['target']  = (
        #    y_i.any_sbo_surg_enc & (y_i.time_to_event_enc < tte_range)).astype(int)
        get_qcut_stats = lambda df: pd.Series(
            [df.shape[0],
             df.target.value_counts().loc[1],
             df[df.target == 1].reset_index()['id'].nunique()],
            index=['num_rows', 'prop_surg', 'num_unique_surg_id'])

        tte_qcut_stats = (
            y_i.dropna(subset=['tte_qcut'])
            .groupby('tte_qcut')
            .apply(get_qcut_stats)
            .round(3))

        hsa_qcut_stats = (
            y_i.dropna(subset=['hsa_qcut'])
            .groupby('hsa_qcut')
            .apply(get_qcut_stats)
            .round(3))

        display(tte_qcut_stats)
        display(hsa_qcut_stats)

        tte_auc_arr[:,i] = (
            y_i.dropna(subset=['tte_qcut'])
            .groupby('tte_qcut')
            .apply(
                lambda df: roc_auc_score(
                    y_score=df['pred'].values, y_true=df['target'].values))
            .values)

        hsa_auc_arr[:,i] = (
            y_i.dropna(subset=['hsa_qcut'])
            .groupby('hsa_qcut')
            .apply(
                lambda df: roc_auc_score(
                    y_score=df['pred'].values, y_true=df['target'].values))
            .values)
        print('Performance over time finished in {}s'.format(round(time.time()-start, 1)))
    tte_auc_df = pd.DataFrame(tte_auc_arr, index=tte_bin_labels)
    hsa_auc_df = pd.DataFrame(hsa_auc_arr, index=hsa_bin_labels)
    return tte_auc_arr, hsa_auc_arr, tte_bin_labels, hsa_bin_labels


# In[5]:


with open('data/processed/result_list_hand.pickle','rb') as f:
    result_list = pickle.load(f)
#img2_full

(lambdas,
        alphas,
        lambda_opt,
        alpha_opt,
        group_fold_dict,
        inner_perf_arr,
        inner_betas_arr,
        betas_df,
        perf_arr,
        perm_imp_df,
        y_train_pred,
        y_test_pred,
        train_means_dict,
        scalers_dict) = result_list


# In[6]:


def compute_pred_ema(df):
    max_hour = df.reset_index().hour_since_adm.max()
    #prev_index = df.index
    out = (df.reset_index(level=[0,1],drop=True)
           .reindex(np.arange(max_hour + 1))
           [['pred']])
    out['pred_ema3'] = (out['pred'].ewm(halflife=3, ignore_na=False).mean())
    out['pred_ema6'] = (out['pred'].ewm(halflife=6, ignore_na=False).mean())
    out['pred_ema24'] = (out['pred'].ewm(halflife=24, ignore_na=False).mean())
    out['pred'] = out['pred'].fillna(method='ffill')
    return out

def fill_in_y(df):
    df['hsa'] = df.index.get_level_values(2)
    # NOTE: this is important because I modified TTE during training
    df['time_to_event_enc'] = df.hsa.transform(lambda x: x.max()-x)
    df['pred'] = df['pred'].fillna(method='ffill', limit=6)
    fold = df.head(1)['fold'].values[0]
    df['fold'] = fold
    surg = df.head(1)['any_sbo_surg_enc'].values[0]
    df['any_sbo_surg_enc'] = surg
    df['target']  = (
        df.any_sbo_surg_enc.astype(bool) & (df.time_to_event_enc < 24)).astype(int)
    return df

test_pred_ema = y_test_pred.groupby(level=[0,1]).apply(compute_pred_ema)


# In[7]:


y_test_ema = pd.concat([y_test_pred.drop('pred',1), test_pred_ema], 1)
y_test_ema = y_test_ema.groupby(level=[0,1]).apply(fill_in_y)


# In[8]:


hsa_bins = np.arange(-1,504,12)
y_test_ema['hsa_cut'] = pd.cut(y_test_ema.hsa, bins=hsa_bins).apply(lambda x: x.right+1).astype(int)
tte_bins = np.arange(-1,504,12)
y_test_ema['tte_cut'] = pd.cut(y_test_ema.time_to_event_enc, bins=tte_bins).apply(lambda x: x.right).astype(int)


# In[9]:


compute_target = lambda df, win: (df.any_sbo_surg_enc.astype(bool)
                                  & (df.time_to_event_enc < win)).astype(int)

y_test_ema['target12']  = compute_target(y_test_ema, 12)
y_test_ema['target24']  = compute_target(y_test_ema, 24)
y_test_ema['target48']  = compute_target(y_test_ema, 48)
y_test_ema['target72']  = compute_target(y_test_ema, 72)


# ## Create Figures

# In[91]:


temp = (y_test_ema.groupby(['hsa']).any_sbo_surg_enc.value_counts()
 .unstack().fillna(0).apply(lambda s: s / s.max()))

plt.figure(figsize=(6,4))
plt.step(temp.index, temp[0], label='Discharge')
plt.step(temp.index, temp[1], label='Surgery')


plt.xlabel('Days since admission')
plt.ylabel('Proportion of patients remaining')
#plt.title('Proportion of patients remaining over time')
plt.grid(True)
#plt.xticks(np.arange(0,504+1,24), np.arange(0,21+1,1))
plt.xticks(np.arange(0,504+1,48), np.arange(0,21+1,2))
plt.legend()
plt.tight_layout()
plt.savefig('figures/survival-curve.pdf');


# In[163]:


temp = (y_test_ema.groupby(['hsa']).any_sbo_surg_enc.value_counts()
 .unstack().fillna(0))
temp = (temp.T[list(np.arange(0, 24*20+2, 2*24))]
 .rename_axis('No. at risk', axis=0)
 .rename(index={0:'Discharge',1: 'Surgery'})
 .rename_axis('', axis=1)
 .astype(int)
 .rename(columns=lambda col: int(col/24)))
temp


# In[207]:


y_test_ema.groupby(level=[0,1]).head(1).groupby('any_sbo_surg_enc')['time_to_event_enc'].median()


# In[194]:


encounters = pd.read_pickle('data/processed/encounters.pickle')


# In[196]:





# In[205]:


pd.concat([y_test_ema.groupby(level=[0,1]).head(1).reset_index(level=2,drop=True),
           encounters], axis=1, join='inner').groupby('any_sbo_surg_enc')[['time_to_event_enc', 'los']].mean()/24


# In[90]:


plt.figure(figsize=(6,4))
ax = (y_test_ema.groupby(level=[0,1]).head(1)
 .assign(time_to_event_enc = lambda df: ((df['time_to_event_enc']+1) // 24))
.groupby(['any_sbo_surg_enc','time_to_event_enc']).size().unstack().T.plot.bar(
    grid=True, width=0.8, rot=0)
)
L = ax.legend(title='')
L.get_texts()[0].set_text('Discharge')
L.get_texts()[1].set_text('Surgery')
#ax.set_title('Ditribution of event times')
plt.xticks(np.arange(0,21+1,2), np.arange(0,21+1,2))
ax.set_xlabel('Days since admission')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('figures/event-hist.pdf');


# # FINAL

# In[92]:


fig, axes = plt.subplots(2,3, figsize=(12,8))
num_bins = y_test_ema.hsa_cut.nunique()
counter = iter(range(num_bins))

def plot_roc(df, c, ax):
    hsa = df.head(1)['hsa_cut'].values[0]
    try:
        fpr, tpr, _ = roc_curve(df.target24, df.pred_ema24)
        label = '{} hrs'.format(hsa)
        ax.plot(fpr,tpr, color=c, label=label)
        ax.grid(True)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        #ax.set_title('ROC curve colored by Hour Since Admission')
        ax.legend(loc='lower right')
    except:
        pass

for i in range(5):
    temp = y_test_ema[y_test_ema.hsa < 120][lambda df: df.fold == i]
    color = iter(plt.cm.Blues(np.linspace(1,0.31,3)))
    for hsa_cut in [24, 72, 120]:
        plot_roc(temp[temp['hsa_cut']==hsa_cut], next(color), axes[i//3,i%3])


plt.axis('off')
plt.tight_layout()
plt.savefig('figures/fold-roc.pdf');


# In[14]:


fig, axes = plt.subplots(2,3, figsize=(12,7),)

def plot_pr(df, c, ax):
    hsa = df.head(1)['hsa_cut'].values[0]
    try:
        precision, recall, _ = precision_recall_curve(df.target24, df.pred)
        #fpr, tpr, _ = roc_curve(df.target24, df.pred_ema)
        label = None
        if (hsa==12) or (hsa==120):
            label = '{} hrs'.format(hsa)
        ax.step(recall,precision, color=c, label=label)
        ax.grid(True)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR curve colored by Hour Since Admission')
        ax.legend(loc='upper right')
    except:
        pass

for i in range(5):
    ax = axes[i//3,i%3]
    temp = y_test_ema[y_test_ema.hsa < 120][lambda df: df.fold == i]
    color = iter(plt.cm.Blues(np.linspace(1,0.25,temp.hsa_cut.nunique())))
    temp.groupby('hsa_cut').apply(lambda df: plot_pr(df, next(color), ax))

plt.tight_layout();


# In[39]:


def try_auc(df, y_col):
    try:
        return roc_auc_score(
            y_score=(df['pred_ema6']).values, y_true=df[y_col].values)
    except:
        return np.nan



ax_shape = (2,2)
fig, axes = plt.subplots(*ax_shape,figsize=(10,7))
axes = axes.reshape(ax_shape)
names = ['target12', 'predict surgery in next 24 hours', 'target48', 'any_sbo_surg_enc']
for i, y_col in enumerate(['target12', 'target24', 'target48', 'any_sbo_surg_enc']):
    try_auc_i = lambda df: try_auc(df, y_col)
    ax_idx = np.unravel_index(i, ax_shape)
    ax = axes[ax_idx]
    (y_test_ema
     #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 240)
     [y_test_ema.hsa < 120]
        .groupby(['hsa_cut','fold'])
        .apply(try_auc_i)
        .unstack().T
        .boxplot(
            #sym='',
                 grid=True, ax=ax,))

    ax.plot(
        pd.Series(
            y_test_ema
            #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 503)
            [lambda df: df.hsa<120]
            .groupby('hsa_cut')
            .apply(try_auc_i)
            .values,
        index = np.arange(1,10+1,1)))

    ax.set_title('Test AUC vs. Hour Since Admission \n({})'.format(names[i]))
    ax.set_xlabel('Hour Since Admission')
    ax.set_ylabel('AUC')
    ax.set_ylim([0.5,1.])
plt.tight_layout();


# # FINAL

# In[185]:


def try_auc(df, y_col):
    try:
        return roc_auc_score(
            y_score=(df['pred_ema6']).values, y_true=df[y_col].values)
    except:
        return np.nan

fig, ax = plt.subplots(1,1,figsize=(6,4))
try_auc_i = lambda df: try_auc(df, 'target24')

auc_over_time = (y_test_ema
 #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 240)
 [y_test_ema.hsa < 120]
    .groupby(['hsa_cut','fold'])
    .apply(try_auc_i)
    .unstack().T)

display(auc_over_time.round(3))

(auc_over_time.boxplot(
    #sym='',
    whis='range', grid=True, ax=ax,))

mean_auc_over_time = pd.Series(
        y_test_ema
        #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 503)
        [lambda df: df.hsa<120]
        .groupby('hsa_cut')
        .apply(try_auc_i)
        .values,
    index = np.arange(1,10+1,1))

display(mean_auc_over_time.to_frame().T.round(3))

ax.plot(mean_auc_over_time)

#ax.set_title('Test AUC vs. Hour Since Admission \n(predict surgery in next 24 hours)')
ax.set_xlabel('Time Since Admission (in hours)')
ax.set_ylabel('AUROC')
ax.set_ylim([0.5,1.])
plt.tight_layout()
plt.savefig('figures/test-auc-hsa-24.pdf');


# In[1]:





# In[21]:


lifelines.utils.concordance_index(
    event_times=y_test_ema.head(1000000)['time_to_event_enc'],
    predicted_scores=-y_test_ema.head(1000000)['pred_ema6'],
    event_observed=y_test_ema.head(1000000)['any_sbo_surg_enc'])


# In[13]:


def c_index(hazard_pred, time_of_event, event_ind):
    """
    betas - (m,1)
    x - (n_samp,m)
    time_of_event - (n_samp,1)
    event_ind - (n_samp)
    Calculate c-index by comparing X @ beta values, and counting
    correct number of rankings.
    """
    observed_event_set   = np.where(event_ind == 1)[0]
    unobserved_event_set = np.where(event_ind == 0)[0]
    correct_ranking_count, num_comparisons = 0, 0

    obs_hazard_preds = hazard_pred[observed_event_set]
    observed_event_times = time_of_event[observed_event_set]

    for obs_haz_pred, event_time in zip(obs_hazard_preds, observed_event_times):
        risk_idx   = np.where(time_of_event >= event_time)[0]
        risk_preds = hazard_pred[risk_idx]
        correct_ranking_count += np.sum(obs_haz_pred > risk_preds)
        num_comparisons += risk_idx.shape[0]

    return correct_ranking_count / num_comparisons


# In[372]:


temp = y_test_ema.head(100000)
c_index(temp['pred_ema6'], temp['time_to_event_enc'], temp['any_sbo_surg_enc'])


# In[122]:




def c_index_over_time(df, y_col='any_sbo_surg_enc'):
    return lifelines.utils.concordance_index(
        event_times=df['time_to_event_enc'],
        # NOTE: uses negative
        predicted_scores=-df['pred_ema6'],
        event_observed=df[y_col])

fig, ax = plt.subplots(1,1,figsize=(6,4))
try_c_i = lambda df: c_index_over_time(df, 'target24')

c_over_time = (y_test_ema
 #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 240)
 [y_test_ema.hsa < 120]
    .groupby(['hsa_cut','fold'])
    .apply(try_c_i)
    .unstack().T)

display(c_over_time.round(3))

(c_over_time.boxplot(
    #sym='',
    whis='range', grid=True, ax=ax,))

mean_c_over_time = pd.Series(
        y_test_ema
        #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 503)
        [lambda df: df.hsa<120]
        .groupby('hsa_cut')
        .apply(try_c_i)
        .values,
    index = np.arange(1,10+1,1))

display(mean_c_over_time.to_frame().T.round(3))

ax.plot(mean_c_over_time)

#ax.set_title('Test AUC vs. Hour Since Admission \n(predict surgery in next 24 hours)')
ax.set_xlabel('Time Since Admission (in hours)')
ax.set_ylabel('C-index')
ax.set_ylim([0.5,1.])
plt.tight_layout()
plt.savefig('figures/test-c-index-hsa-24.pdf');


# # FINAL

# In[133]:


def try_auc(df, y_col):
    try:
        return roc_auc_score(
            y_score=(df['pred_ema6']).values, y_true=df[y_col].values)
    except:
        return np.nan

fig, ax = plt.subplots(1,1,figsize=(6,4))
try_auc_i = lambda df: try_auc(df, 'any_sbo_surg_enc')

auc_over_time = (y_test_ema
 #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 240)
 [y_test_ema.hsa < 120]
    .groupby(['hsa_cut','fold'])
    .apply(try_auc_i)
    .unstack().T)

display(auc_over_time.round(3))

(auc_over_time.boxplot(
    #sym='',
    whis='range', grid=True, ax=ax,))

mean_auc_over_time = pd.Series(
        y_test_ema
        #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 503)
        [lambda df: df.hsa<120]
        .groupby('hsa_cut')
        .apply(try_auc_i)
        .values,
    index = np.arange(1,10+1,1))

display(mean_auc_over_time.to_frame().T.round(3))

ax.plot(mean_auc_over_time)

#ax.set_title('Test AUC vs. Hour Since Admission \n(predict surgery in next 24 hours)')
ax.set_xlabel('Time Since Admission (in hours)')
ax.set_ylabel('AUROC')
ax.set_ylim([0.5,1.])
plt.tight_layout()
plt.savefig('figures/test-auc-hsa-anysbo.pdf');


# In[97]:




def c_index_over_time(df, y_col='any_sbo_surg_enc'):
    return lifelines.utils.concordance_index(
        event_times=df['time_to_event_enc'],
        # NOTE: uses negative
        predicted_scores=-df['pred_ema6'],
        event_observed=df[y_col])

fig, ax = plt.subplots(1,1,figsize=(6,4))
try_c_i = lambda df: c_index_over_time(df, 'any_sbo_surg_enc')

c_over_time = (y_test_ema
 #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 240)
 [y_test_ema.hsa < 120]
    .groupby(['hsa_cut','fold'])
    .apply(try_c_i)
    .unstack().T)

display(c_over_time.round(3))

(c_over_time.boxplot(
    #sym='',
    whis='range', grid=True, ax=ax,))

mean_c_over_time = pd.Series(
        y_test_ema
        #.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 503)
        [lambda df: df.hsa<120]
        .groupby('hsa_cut')
        .apply(try_c_i)
        .values,
    index = np.arange(1,10+1,1))

display(mean_c_over_time.to_frame().T.round(3))

ax.plot(mean_c_over_time)

#ax.set_title('Test AUC vs. Hour Since Admission \n(predict surgery in next 24 hours)')
ax.set_xlabel('Time Since Admission (in hours)')
ax.set_ylabel('C-index')
ax.set_ylim([0.5,1.])
plt.tight_layout()
plt.savefig('figures/test-c-index-hsa-any.pdf');


# # FINAL

# In[132]:



#plt.figure(figsize=(6,4))
try_auc_i = lambda df: try_auc(df, 'any_sbo_surg_enc')

auc_tte = (y_test_ema[y_test_ema.time_to_event_enc < 72]
    .groupby(['fold', 'tte_cut']).apply(try_auc_i).unstack()
           )
auc_tte = auc_tte.rename(columns=dict(zip(auc_tte, auc_tte.columns + 1)))

display(auc_tte.round(3))

ax = auc_tte.boxplot(figsize=(6,4), whis='range')

auc_tte_mean = pd.Series(y_test_ema[y_test_ema.time_to_event_enc < 72]
    .groupby('tte_cut')
    .apply(try_auc_i).values, index = np.arange(1,6+1,1))

display(auc_tte_mean.to_frame().T.round(3))

ax.plot(auc_tte_mean)
#ax.set_title('Test AUC vs. Time to Event')
ax.set_xlabel('Time to Event')
ax.set_ylabel('AUROC')
plt.xticks(np.arange(1,6+1,1),np.arange(12,121,12))
plt.tight_layout()
plt.savefig('figures/test-auc-tte.pdf');


# In[125]:



plt.figure(figsize=(6,4))
auc_tte = (y_test_ema[y_test_ema.time_to_event_enc < 72]
    .groupby(['fold', 'tte_cut']).apply(try_c_i).unstack()
           )
auc_tte = auc_tte.rename(columns=dict(zip(auc_tte, auc_tte.columns + 1)))

display(auc_tte.round(3))

ax = auc_tte.boxplot(figsize=(6,4), whis='range')

auc_tte_mean = pd.Series(y_test_ema[y_test_ema.time_to_event_enc < 72]
    .groupby('tte_cut')
    .apply(try_c_i).values, index = np.arange(1,6+1,1))

display(auc_tte_mean.to_frame().T.round(3))

ax.plot(auc_tte_mean)
#ax.set_title('Test AUC vs. Time to Event')
ax.set_xlabel('Time to Event')
ax.set_ylabel('C-index')
plt.xticks(np.arange(1,6+1,1),np.arange(12,121,12))
plt.tight_layout()
plt.savefig('figures/test-c-index-tte.pdf');


# # FINAL

# In[130]:


fig, ax = plt.subplots(1,1,figsize=(4,4))

def get_mean_pr(y_test_ema, c, label):
    tprs = []
    mean_fpr = np.linspace(0.005, 1, 200)

    for i in range(5):
        temp = y_test_ema[y_test_ema['fold'] == i]
        fpr, tpr, _ = roc_curve(temp['target24'], temp['pred_ema6'])
        f = interp1d(fpr, tpr)
        tprs.append(f(mean_fpr))

    mean_tpr = np.mean(tprs, axis=0)
    ax.step(mean_fpr,mean_tpr, color=c, label=label)
    min_tpr = np.min(tprs, axis=0)
    max_tpr = np.max(tprs, axis=0)
    #ax.fill_between(mean_fpr,min_tpr,max_tpr, color=c,alpha=0.5, label=label)
    ax.grid(True)
    ax.set_xlabel('FPR (1-specificity)')
    ax.set_ylabel('TPR (sensitivity)')
    ax.set_xticks(np.linspace(0,1,6))
    ax.legend(loc='lower right')
    #ax.set_title('ROC curve for 24 hr prediction window \n (colored by hour since admission)')

# 10 b/c 12 - 120 by increments of 12
color = iter(plt.cm.Blues(np.linspace(1,0.31, 3)))
for i in [24,72,120]:
    label = '{} hours'.format(i)
    get_mean_pr(y_test_ema[y_test_ema['hsa_cut']==i], next(color), label)

plt.tight_layout()
plt.savefig('figures/mean-auc-24.pdf');


# # FINAL

# In[131]:


fig, ax = plt.subplots(1,1,figsize=(4,4))

def get_mean_pr(y_test_ema, c, label):
    tprs = []
    mean_fpr = np.linspace(0.005, 1, 200)

    for i in range(5):
        temp = y_test_ema[y_test_ema['fold'] == i]
        fpr, tpr, _ = roc_curve(temp['any_sbo_surg_enc'], temp['pred_ema6'])
        f = interp1d(fpr, tpr)
        tprs.append(f(mean_fpr))

    mean_tpr = np.mean(tprs, axis=0)
    ax.step(mean_fpr,mean_tpr, color=c, label=label)
    min_tpr = np.min(tprs, axis=0)
    max_tpr = np.max(tprs, axis=0)
    #ax.fill_between(mean_fpr,min_tpr,max_tpr, color=c,alpha=0.5, label=label)
    ax.grid(True)
    ax.set_xlabel('FPR (1-specificity)')
    ax.set_ylabel('TPR (sensitivity)')
    ax.set_xticks(np.linspace(0,1,6))
    ax.legend(loc='lower right')
    #ax.set_title('ROC curve for 24 hr prediction window \n (colored by hour since admission)')

# 10 b/c 12 - 120 by increments of 12
color = iter(plt.cm.Blues(np.linspace(1,0.31, 3)))
for i in [24,72,120]:
    label = '{} hours'.format(i)
    get_mean_pr(y_test_ema[y_test_ema['hsa_cut']==i], next(color), label)

plt.tight_layout()
plt.savefig('figures/mean-auc-anysbo.pdf');


# In[22]:


from scipy import interp
from scipy.interpolate import interp1d

fig, ax = plt.subplots(1,1,figsize=(5,5))

def get_mean_pr(y_test_ema, c, label):
    precisions = []
    mean_recall = np.linspace(0.005, 1, 200)

    for i in range(5):
        temp = y_test_ema[y_test_ema['fold'] == i]
        precision, recall, _ = precision_recall_curve(temp['target24'], temp['pred_ema6'])
        f = interp1d(recall, precision)
        precisions.append(f(mean_recall))

    mean_precision = np.mean(precisions, axis=0)
    ax.step(mean_recall,mean_precision, color=c, label=label)
    ax.grid(True)
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Positive predictive value')
    ax.legend(loc='upper right')
    ax.set_title('Predict no SBO surgery \n (colored by hour since admission)')

# 10 b/c 12 - 120 by increments of 12
color = iter(plt.cm.Blues(np.linspace(1,0.31, 10)))
for i in range(1,10+1):
    label = None
    if (i==1) or (i==10):
        label = f'{i*12} hours'
    get_mean_pr(y_test_ema[y_test_ema['hsa_cut']==12*i], next(color), label)

plt.tight_layout();


# # FINAL

# ## NPV

# In[104]:


fig, ax = plt.subplots(1,1,figsize=(5,5))

def get_mean_pr(y_test_ema, c, label):
    precisions = []
    mean_recall = np.linspace(0.005, 1, 200)

    for i in range(5):
        temp = y_test_ema[y_test_ema['fold'] == i]
        precision, recall, _ = precision_recall_curve(1-temp['any_sbo_surg_enc'], -temp['pred_ema6'])
        f = interp1d(recall, precision)
        precisions.append(f(mean_recall))

    mean_precision = np.mean(precisions, axis=0)
    ax.step(mean_recall,mean_precision,
            color=c,
            #c=list(objective_function(mean_precision,mean_recall,q=10)),
            label=label)
    ax.grid(True)
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Negative predictive value')
    ax.legend(loc='lower left')
    #ax.set_title('Predict no SBO surgery \n (colored by hour since admission)')

# 10 b/c 12 - 120 by increments of 12
color = iter(plt.cm.Blues(np.linspace(1,0.31, 3)))
for i in [24,72,120]:
    label = '{} hours'.format(i)
    get_mean_pr(y_test_ema[y_test_ema['hsa_cut']==i], next(color), label)

plt.tight_layout()
plt.savefig('figures/neg-pr-anysbo.pdf');


# # FINAL

# In[105]:


fig, ax = plt.subplots(1,1,figsize=(5,5))

def get_mean_pr(y_test_ema, c, label):
    precisions = []
    mean_recall = np.linspace(0.005, 1, 200)

    for i in range(5):
        temp = y_test_ema[y_test_ema['fold'] == i]
        precision, recall, _ = precision_recall_curve(1-temp['target24'], -temp['pred_ema6'])
        f = interp1d(recall, precision)
        precisions.append(f(mean_recall))

    mean_precision = np.mean(precisions, axis=0)
    ax.step(mean_recall,mean_precision,
            color=c,
            #c=list(objective_function(mean_precision,mean_recall,q=10)),
            label=label)
    ax.grid(True)
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Negative predictive value')
    ax.legend(loc='lower left')
    plt.ylim([0.9,1.])
    #ax.set_title('Predict no SBO surgery \n (colored by hour since admission)')

# 10 b/c 12 - 120 by increments of 12
color = iter(plt.cm.Blues(np.linspace(1,0.31,3)))
for i in [24,72,120]:
    label = '{} hours'.format(i)
    get_mean_pr(y_test_ema[y_test_ema['hsa_cut']==i], next(color), label)

plt.tight_layout()
plt.savefig('figures/neg-pr-24-scale.pdf');


# In[106]:


fig, ax = plt.subplots(1,1,figsize=(5,5))

def get_mean_pr(y_test_ema, c, label):
    precisions = []
    mean_recall = np.linspace(0.005, 1, 200)

    for i in range(5):
        temp = y_test_ema[y_test_ema['fold'] == i]
        precision, recall, _ = precision_recall_curve(1-temp['target24'], -temp['pred_ema6'])
        f = interp1d(recall, precision)
        precisions.append(f(mean_recall))

    mean_precision = np.mean(precisions, axis=0)
    ax.step(mean_recall,mean_precision,
            color=c,
            #c=list(objective_function(mean_precision,mean_recall,q=10)),
            label=label)
    ax.grid(True)
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Negative predictive value')
    ax.legend(loc='lower left')
    #plt.ylim([0.9,1.])
    #ax.set_title('Predict no SBO surgery \n (colored by hour since admission)')

# 10 b/c 12 - 120 by increments of 12
color = iter(plt.cm.Blues(np.linspace(1,0.31,3)))
for i in [24,72,120]:
    label = '{} hours'.format(i)
    get_mean_pr(y_test_ema[y_test_ema['hsa_cut']==i], next(color), label)

plt.tight_layout()
plt.savefig('figures/neg-pr-24-noscale.pdf');


# # New discharge task strategy

# In[ ]:





# In[22]:


fig, ax = plt.subplots(1,1,figsize=(5,5))

def get_mean_pr(y_test_ema, c, label):
    precisions = []
    mean_recall = np.linspace(0.005, 1, 200)

    for i in range(5):
        temp = y_test_ema[y_test_ema['fold'] == i]
        precision, recall, _ = precision_recall_curve(1-temp['any_sbo_surg_enc'], -temp['pred_ema6'])
        f = interp1d(recall, precision)
        precisions.append(f(mean_recall))

    mean_precision = np.mean(precisions, axis=0)
    ax.step(mean_recall,mean_precision,
            color=c,
            #c=list(objective_function(mean_precision,mean_recall,q=10)),
            label=label)
    ax.grid(True)
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Negative predictive value')
    ax.legend(loc='lower left')
    #ax.set_title('Predict no SBO surgery \n (colored by hour since admission)')

# 10 b/c 12 - 120 by increments of 12
color = iter(plt.cm.Blues(np.linspace(1,0.31, 3)))
for i in [24,72,120]:
    label = '{} hours'.format(i)
    get_mean_pr(y_test_ema[y_test_ema['hsa_cut']==i], next(color), label)

plt.tight_layout()
plt.savefig('figures/neg-pr-anysbo.pdf');


# In[85]:





# In[10]:


from sklearn.metrics import precision_score, recall_score, confusion_matrix


# In[11]:


# do run
y_cp = y_test_ema.copy()


# In[179]:


# dont run
#/ pred.std()
#y_cp['pred_ema6'] = y_cp.groupby('hsa_cut')['pred_ema6'].transform(lambda pred: (pred - pred.mean()) )


# In[207]:


plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
y_cp[y_cp['hsa']<120][lambda df: df['any_sbo_surg_enc'] == 0].groupby('hsa_cut')['pred_ema6'].hist(bins=20)

plt.subplot(1,2,2)
y_cp[y_cp['hsa']<120][lambda df: df['any_sbo_surg_enc'] == 1].groupby('hsa_cut')['pred_ema6'].hist(bins=20)

plt.tight_layout();


# In[208]:




fig, axes = plt.subplots(1,2, figsize=(8,4), sharey=True)
#plt.subplot(1,2,1)
surg_dist = (
    y_cp[y_cp['hsa']<240][lambda df: df['any_sbo_surg_enc'] == 0]
    .groupby('hsa_cut')['pred_ema6']
    .agg(['mean', 'std']))
#.hist(bins=20)

hsa_cuts = np.arange(12,240+12,12)
axes[0].plot(hsa_cuts, surg_dist['mean'])
axes[0].fill_between(hsa_cuts, surg_dist['mean']- 2*surg_dist['std'],
                     surg_dist['mean'] + 2*surg_dist['std'], alpha=0.1)

#plt.subplot(1,2,2)
non_surg_dist = (
    y_cp[y_cp['hsa']<240][lambda df: df['any_sbo_surg_enc'] == 1]
    .groupby('hsa_cut')['pred_ema6']
    .agg(['mean', 'std']))

axes[1].plot(hsa_cuts, non_surg_dist['mean'])
axes[1].fill_between(hsa_cuts, non_surg_dist['mean']- 2*non_surg_dist['std'],
                     non_surg_dist['mean'] + 2*non_surg_dist['std'], alpha=0.1)
plt.tight_layout();


# In[ ]:





# In[81]:


plt.hist(y_cp['pred_ema6'], bins=100);


# In[185]:


pd.concat(map(lambda thresh: (y_cp['pred_ema6'] > thresh).rename(thresh), [0, 0.5]), axis=1)


# In[935]:


def agg_early_disch(s):
    print(s)
    idx = s[s==True].first_valid_index()
    print(idx)
    try:
        print(idx[2])
        return idx[2]
    except:
        return None


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[12]:


# do run
step = 0.02
thresh_list = list(np.arange(-1, 1+step, step).round(4))


# In[13]:


def scan_thresh(df):
    global thresh_array_cp
    thresh_array = np.array(thresh_list).round(2)
    thresh_array_cp = np.copy(thresh_array)
    hit_thresh_idx = np.array([np.nan]*len(thresh_list))
    def compare_risk(risk):
        global thresh_array_cp
        hits = risk.values[0] < thresh_array_cp
        #print(risk)
        #print(risk.values[0], thresh_array_cp, hits)
        if hits.any():
            new_hit_idx = np.where(hits)[0]
            curr_time = risk.name[2]
            hit_thresh_idx[new_hit_idx] = curr_time
            thresh_array_cp = thresh_array_cp[:new_hit_idx.min()]
            #print('HIT!')
            #print(new_hit_idx)
            #print(curr_time)
            #print(hit_thresh_idx)
            #print(thresh_array_cp)

    _ = df[['pred_ema6']].apply(compare_risk, axis=1)

    return pd.Series(hit_thresh_idx, index=thresh_array)


# In[611]:


get_ipython().run_cell_magic('time', '', '#thresh_grid = y_cp.head(10000).groupby(level=[0,1]).head(5).head(30).groupby(level=[0,1]).apply(scan_thresh)')


# In[ ]:





# In[14]:


get_ipython().run_cell_magic('time', '', "thresh_grid = y_cp.groupby(level=[0,1]).apply(scan_thresh)\nbase = y_cp.groupby(level=[0,1]).head(1).reset_index(\n        level=2,drop=True)[['any_sbo_surg_enc', 'time_to_event_enc']]\n\n\ntime_df = pd.concat([base, thresh_grid], axis=1)\nbool_df = pd.concat([base, thresh_grid.notna()], axis=1)")


# In[32]:


time_df.head()


# In[ ]:





# In[15]:



def early_disch_time_info(df):
    start = time.time()
    df = df.copy()

    bool_cols = pd.concat(map(lambda thresh: (df['pred_ema6'] < thresh).rename(thresh), thresh_list), axis=1)
    print('bool {}s'.format(round(time.time()-start)))
    #early_disch_time = bool_cols.groupby(level=[0,1]).apply(lambda df: df.apply(agg_early_disch))


    early_disch_time = bool_cols.apply(lambda s: s[s==True].groupby(level=[0,1]).apply(lambda df: df.first_valid_index()))


    #early_disch_time = bool_cols.apply(lambda s: s[s==True])

#     #last_index = None
#     bool_cols_cp = bool_cols.copy()
#     res = []
#     for col in bool_cols.columns:
# #         if last_index is not None:
# #             print(s.shape, last_index.shape)
# #             s = s[last_index]
#         s = bool_cols[col]
#         #print(s.shape)

#         #print('use last', s.shape)
#         new_index = s==True
#         #print(new_index.shape)
#         s = s[new_index]
#         #print(s.shape)
#         #last_index = new_index
#         bool_cols = bool_cols[new_index]
#         res.append(s.groupby(level=[0,1]).apply(lambda df: df.first_valid_index()))
#     early_disch_time = pd.concat(res, axis=1)

    print('agg early disch {}s'.format(round(time.time()-start)))
    any_disch = early_disch_time.notna()
    #.rename(columns = lambda col: col + '_bool')
    base = df.groupby(level=[0,1]).head(1).reset_index(
        level=2,drop=True)[['any_sbo_surg_enc', 'time_to_event_enc']]
    #.set_index(['any_sbo_surg_enc', 'time_to_event_enc'], append=True)
    #pd.concat([df.groupby(level=[0,1]).head(1).reset_index(level=2,drop=True),
    #                  early_disch_time, any_disch], 1)

    return pd.concat([base, early_disch_time], 1), pd.concat([base, any_disch], 1)


# In[16]:


def breakdown_confusion(y_true, y_pred):
    conf_mtx = confusion_matrix(y_true, y_pred)
    tn = conf_mtx[0,0]
    fp = conf_mtx[0,1]
    fn = conf_mtx[1,0]
    tp = conf_mtx[1,1]
    return tn, fp, fn, tp

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = breakdown_confusion(y_true, y_pred)
    return tn / (tn + fp)


# In[665]:





# In[17]:


def compute_scores(time_df, bool_df):

    scores = []
    times = []
    times_na = []

    for thresh in thresh_list:

        npv = precision_score(y_true=1-bool_df['any_sbo_surg_enc'],
                         y_pred = bool_df[thresh].astype(int))

        # negative case -- percent ; 1-fpr=tnr=specificity for neg

        # tn = # correctly not discharged
        # fp = # accidentally discharged
        # fn = # accidentally not discharged
        # tp = # correctly discharged
        tn, fp, fn, tp = breakdown_confusion(
            y_true=bool_df['any_sbo_surg_enc'],
            y_pred = 1-bool_df[thresh].astype(int))

        # specificity for neg task
        ss = tn / (tn + fp)

#         # npv for positive case
#         ss = specificity_score(y_true=1-bool_df['any_sbo_surg_enc'],
#                           y_pred = bool_df[thresh].astype(int))
#         conf_mtx = confusion


        # This is specificity when defined for the positive case???
        # recall = tp / (tp + fn)
        # recall fn on neg case is computing tn / (tn + fp) ?????
        rs = recall_score(y_true=1-bool_df['any_sbo_surg_enc'],
                          y_pred = bool_df[thresh].astype(int))
        rs = tp / (tp + fn)
        scores += [[rs, npv, ss]]


        df = time_df[time_df['any_sbo_surg_enc']==0].copy().dropna(subset=[thresh])
        #print(thresh, df.shape, df[df[thresh]==0].shape)
        df = df['time_to_event_enc'] - df[thresh]
        times += [[df.mean(), df.sum()]]

        # Filter out surg ppl, only "save time" for discharge pop
        df = time_df[time_df['any_sbo_surg_enc']==0].copy()
        df[thresh] = df[thresh].fillna(df['time_to_event_enc'])
        #print(df.shape, (df[df[thresh]==0].shape))
        #df = time_df.fillna(subset=[thresh])
        df = df['time_to_event_enc'] - df[thresh]
        times_na += [[df.mean(), df.sum()]]

    return np.array(scores), np.array(times), np.array(times_na)


# In[18]:


def compute_scores(time_df, bool_df):

    scores = []
    times = []
    times_na = []

    for thresh in thresh_list:

        npv = precision_score(y_true=1-bool_df['any_sbo_surg_enc'],
                         y_pred = bool_df[thresh].astype(int))

        # negative case -- percent ; 1-fpr=tnr=specificity for neg

        # tn = # correctly not discharged
        # fp = # accidentally discharged
        # fn = # accidentally not discharged
        # tp = # correctly discharged
        tn, fp, fn, tp = breakdown_confusion(
            y_true=1-bool_df['any_sbo_surg_enc'],
            y_pred =bool_df[thresh].astype(int))

        # specificity for neg task
        ss = tn / (tn + fp)

#         # npv for positive case
#         ss = specificity_score(y_true=1-bool_df['any_sbo_surg_enc'],
#                           y_pred = bool_df[thresh].astype(int))
#         conf_mtx = confusion


        # This is specificity when defined for the positive case???
        # recall = tp / (tp + fn)
        # recall fn on neg case is computing tn / (tn + fp) ?????
        rs = recall_score(y_true=1-bool_df['any_sbo_surg_enc'],
                          y_pred = bool_df[thresh].astype(int))
        rs = tp / (tp + fn)
        scores += [[rs, npv, ss]]


        df = time_df[time_df['any_sbo_surg_enc']==0].copy().dropna(subset=[thresh])
        #print(thresh, df.shape, df[df[thresh]==0].shape)
        df = df['time_to_event_enc'] - df[thresh]
        times += [[df.mean(), df.sum()]]

        # Filter out surg ppl, only "save time" for discharge pop
        df = time_df[time_df['any_sbo_surg_enc']==0].copy()
        df[thresh] = df[thresh].fillna(df['time_to_event_enc'])
        #print(df.shape, (df[df[thresh]==0].shape))
        #df = time_df.fillna(subset=[thresh])
        df = df['time_to_event_enc'] - df[thresh]
        times_na += [[df.mean(), df.sum()]]

    return np.array(scores), np.array(times), np.array(times_na)


# In[19]:


disch_scores, disch_times, disch_times_wna = compute_scores(time_df, bool_df)


# In[20]:


tn, fp, fn, tp = breakdown_confusion(y_true=1-bool_df['any_sbo_surg_enc'],
                          y_pred = bool_df[-0.18].astype(int))
print("tn={}, fp={}, fn={}, tp={}".format(tn, fp, fn, tp))
print("actually discharged = {}".format(tp+fn))
print("actually surgery = {}".format(tn+fp))
print("predicted discharged = {}".format(tp+fp))
print("predicted no discharge = {}".format(tn+fn))
# fp / tn + fp
# 1 - tp / tp + fn = 1 - recall


# In[21]:


thresh_list[41]


# In[26]:


thresh_num=22
print('Threshold = {}, Specificity = {:.3f}, dPPV = {:.3f}, dRecall = {:.3f}'.format(
    thresh_list[thresh_num],
    disch_scores[thresh_num,2],
    disch_scores[thresh_num,1],
    disch_scores[thresh_num,0]))


# In[ ]:


np.random


# In[ ]:


thresh_num=41
print('Threshold = {}, Specificity = {:.3f}, dPPV = {:.3f}, dRecall = {:.3f}'.format(
    thresh_list[thresh_num],
    disch_scores[thresh_num,2],
    disch_scores[thresh_num,1],
    disch_scores[thresh_num,0]))


# In[28]:


disch_times_wna[thresh_num,0]


# In[184]:


plt.figure(figsize=(5.25,5))

plt.step( 1-disch_scores[:,2], disch_times_wna[:,0], label='axis1',)
plt.ylabel('Expected time saved per patient')
plt.xlabel('Incorrect Discharge Rate')
plt.grid(True)
plt.scatter(1-disch_scores[thresh_num,2], disch_times_wna[thresh_num,0], c='red', s=70)

ax = plt.gca()
ax.set_xlim([0-0.01, 0.3+0.01])
ax.set_xticks(np.linspace(0,0.3, 7))
ax.set_ylim([0-5, 120 +5])
xt = ax.get_xticks()
#xt = np.append(xt, 0.094)
xtl = [round(x,2) for x in xt]
#xtl[-1] = r'$\mathbf{r}$'
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
#ax.get_xticklabels()[-1].set_color("red")


plt.tight_layout()
plt.savefig('figures/specificity-vs-time.pdf');


# In[ ]:





# In[182]:


plt.figure(figsize=(5.25,5))

plt.step(disch_scores[:,0], disch_scores[:,1])
plt.xlabel('True Discharge Rate')
plt.ylabel('Discharge Predictive Value')
plt.scatter(disch_scores[thresh_num,0], disch_scores[thresh_num,1], c='red', s=70)
plt.grid(True)

plt.tight_layout()
plt.savefig('figures/discharge-pr-curve.pdf');


# In[ ]:


def disch_proc(df):
    df = df.copy()
    df.set_index('any_sbo_surg_enc')
    #thresh = np.array([0,0.5])
    thresh_df = pd.concat(map(lambda thresh: (df['pred_ema6'] > thresh).any(), [0, 0.5]), axis=1)
    #thresh_df = (df['pred_ema6'] > thresh)
    #df = pd.concat([df, thresh_df], axis=1)
    #df['disch_ever'] = df.groupby(level=1).transform(lambda x: x.any())
    display(df)
    #df2 = df.groupby(level=1).head(1)
    #return precision_score(y_true=1-df2['any_sbo_surg_enc'], y_pred=df2['disch_ever'])


# In[197]:


y_cp.head(100)[lambda df: df['hsa_cut'] < 121].groupby('hsa_cut').apply(disch_proc)


# In[ ]:





# In[ ]:





# ### how many surgeries at any point?

# In[211]:


y_cp.groupby(level=[0,1]).tail(1).groupby(['hsa_cut', 'fold'])['target24'].value_counts().unstack()


# # SURGERY PROCEDURE

# In[356]:


step = 0.01
thresh_list = list(np.arange(-1, 1.5+step, step).round(4))


# In[955]:


def agg_early_disch(s):
    print(s)
    idx = s[s==True].first_valid_index()
    print(idx)
    try:
        print(idx[2])
        return idx[2]
    except:
        return None


# In[205]:


np.zeros(4)


# In[357]:






def scan_thresh(df):
    global thresh_array_cp, idx_offset, hit_thresh_count
    thresh_array = np.array(thresh_list).round(4)
    thresh_array_cp = np.copy(thresh_array)
    hit_thresh_idx = np.array([np.nan]*len(thresh_list))
    hit_thresh_count = np.zeros(len(thresh_list))
    # removing thresholds from list so changes how to index into storage list, exclusive so start at -1 instead of 0
    idx_offset = -1
    hsa_mod_fn = lambda t: np.exp(-t/24)
    #print('new person')
    def compare_risk(risk):
        global thresh_array_cp, idx_offset, hit_thresh_count
        curr_time = risk.name[2]

        hits = risk.values[0] > thresh_array_cp
        #+ hsa_mod_fn(curr_time)
        #print(risk)

        num_hits = 24

        if hits.any():
            new_hit_idx = np.where(hits)[0]
            #print('HIT!')

            #print(idx_offset)
            #hit_thresh_idx[idx_offset + 1 + new_hit_idx] = curr_time
            hit_thresh_idx[idx_offset + 1 + new_hit_idx] = curr_time
            thresh_array_cp = thresh_array_cp[new_hit_idx.max()+1:]
            #hit_thresh_count = hit_thresh_count[new_hit_idx.max()+1:]
            idx_offset = idx_offset + new_hit_idx.max() + 1
            #print(risk.values[0], thresh_array_cp, hits)
            #print('super HIT!')
            ##print(new_hit_idx,idx_offset)
            #print(curr_time)


#         if hits.any():
#             new_hit_idx = np.where(hits)[0]
#             #print('HIT!')

#             ##print('test', hit_thresh_count, idx_offset, new_hit_idx)
#             hit_thresh_count[new_hit_idx] = (
#                 hit_thresh_count[new_hit_idx] + 1)
#             ##print(hit_thresh_count, (hit_thresh_count == 3).any())

#             if (hit_thresh_count == num_hits).any():

#                 multi_hit_idx = np.where(hit_thresh_count == num_hits)[0]
#                 ##print('idxs', new_hit_idx, multi_hit_idx)

#                 #print(idx_offset)
#                 #hit_thresh_idx[idx_offset + 1 + new_hit_idx] = curr_time
#                 hit_thresh_idx[idx_offset + 1 + multi_hit_idx] = curr_time
#                 thresh_array_cp = thresh_array_cp[multi_hit_idx.max()+1:]
#                 hit_thresh_count = hit_thresh_count[multi_hit_idx.max()+1:]
#                 idx_offset = idx_offset + multi_hit_idx.max() + 1
#                 #print(risk.values[0], thresh_array_cp, hits)
#                 #print('super HIT!')
#                 ##print(new_hit_idx,idx_offset)
#                 #print(curr_time)
#                 ##print(hit_thresh_idx)
#                 #print(thresh_array_cp)

    _ = df[['pred_ema6']].apply(compare_risk, axis=1)

    return pd.Series(hit_thresh_idx, index=thresh_array)


# In[812]:


y_cp.loc[('42164',107894400)]


# In[308]:


#%%time
thresh_grid = y_cp.loc[['42164','2374']].groupby(level=[0,1]).apply(scan_thresh)


# In[975]:


thresh_grid.iloc[:,50:]


# In[318]:


#%%time
thresh_grid = y_cp.head(10000).groupby(level=[0,1]).head(100).head(3000).groupby(level=[0,1]).apply(scan_thresh)


# In[ ]:





# In[358]:


get_ipython().run_cell_magic('time', '', "\n# NOTE cutting off at 120\n#(df['hsa'] > 24) &\nnum_hits = 24\nthresh_grid = (\n    y_cp\n    #.groupby(level=[0,1])\n    #.filter(lambda df: df['hsa'].max() > num_hits)\n    [lambda df: (df['hsa'] < 240)]\n    .groupby(level=[0,1])\n    .apply(scan_thresh))\nbase = (\n    y_cp\n    #.groupby(level=[0,1])\n    #.filter(lambda df: df['hsa'].max() > num_hits)\n    [lambda df: (df['hsa'] < 240)]\n    .groupby(level=[0,1]).head(1)\n    .reset_index(level=2,drop=True)\n    [['any_sbo_surg_enc', 'time_to_event_enc']])\n\n\ntime_df = pd.concat([base, thresh_grid], axis=1)\nbool_df = pd.concat([base, thresh_grid.notna()], axis=1)")


# In[316]:


time_df.head()


# In[274]:


time_df[0.4].hist(bins=20)


# In[1040]:


time_df.iloc[:16,2:].T.plot(legend=False)
#plt.xlim([0.2,1])


# In[ ]:





# In[41]:



def early_disch_time_info(df):
    start = time.time()
    df = df.copy()

    bool_cols = pd.concat(map(lambda thresh: (df['pred_ema6'] < thresh).rename(thresh), thresh_list), axis=1)
    print('bool {}s'.format(round(time.time()-start)))
    #early_disch_time = bool_cols.groupby(level=[0,1]).apply(lambda df: df.apply(agg_early_disch))


    early_disch_time = bool_cols.apply(lambda s: s[s==True].groupby(level=[0,1]).apply(lambda df: df.first_valid_index()))


    #early_disch_time = bool_cols.apply(lambda s: s[s==True])

#     #last_index = None
#     bool_cols_cp = bool_cols.copy()
#     res = []
#     for col in bool_cols.columns:
# #         if last_index is not None:
# #             print(s.shape, last_index.shape)
# #             s = s[last_index]
#         s = bool_cols[col]
#         #print(s.shape)

#         #print('use last', s.shape)
#         new_index = s==True
#         #print(new_index.shape)
#         s = s[new_index]
#         #print(s.shape)
#         #last_index = new_index
#         bool_cols = bool_cols[new_index]
#         res.append(s.groupby(level=[0,1]).apply(lambda df: df.first_valid_index()))
#     early_disch_time = pd.concat(res, axis=1)

    print('agg early disch {}s'.format(round(time.time()-start)))
    any_disch = early_disch_time.notna()
    #.rename(columns = lambda col: col + '_bool')
    base = df.groupby(level=[0,1]).head(1).reset_index(
        level=2,drop=True)[['any_sbo_surg_enc', 'time_to_event_enc']]
    #.set_index(['any_sbo_surg_enc', 'time_to_event_enc'], append=True)
    #pd.concat([df.groupby(level=[0,1]).head(1).reset_index(level=2,drop=True),
    #                  early_disch_time, any_disch], 1)

    return pd.concat([base, early_disch_time], 1), pd.concat([base, any_disch], 1)


# In[42]:


def breakdown_confusion(y_true, y_pred):
    conf_mtx = confusion_matrix(y_true, y_pred)
    tn = conf_mtx[0,0]
    fp = conf_mtx[0,1]
    fn = conf_mtx[1,0]
    tp = conf_mtx[1,1]
    return tn, fp, fn, tp

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = breakdown_confusion(y_true, y_pred)
    return tn / (tn + fp)


# In[281]:


tn, fp, fn, tp = breakdown_confusion(y_true=bool_df['any_sbo_surg_enc'],
                          y_pred = bool_df[0.4].astype(int))


# In[282]:


tn, fp, fn, tp


# In[348]:


def compute_scores(time_df, bool_df):

    all_scores = []
    all_times = []
    all_times_na = []

    time_dfs = [time_df[( cutoff[0] <= time_df['time_to_event_enc'])
                        & (time_df['time_to_event_enc'] < cutoff[1])]
                for cutoff
                in [(0, 24), (24,72), (72,120), (120,240)]]

    bool_dfs = [bool_df[( cutoff[0] <= bool_df['time_to_event_enc'])
                        & (bool_df['time_to_event_enc'] < cutoff[1])]
                for cutoff
                in [(0, 24), (24,72), (72,120), (120,240)]]

    for time_df, bool_df in zip(time_dfs, bool_dfs):

        scores = []
        times = []
        times_na = []

        for thresh in thresh_list:

            ppv = precision_score(y_true=bool_df['any_sbo_surg_enc'],
                              y_pred = bool_df[thresh].fillna(False).astype(int))

            # negative case -- percent ; 1-fpr=tnr=specificity for neg

            # tn = # correctly not discharged
            # fp = # accidentally discharged
            # fn = # accidentally not discharged
            # tp = # correctly sent f
            tn, fp, fn, tp = breakdown_confusion(
                y_true=bool_df['any_sbo_surg_enc'],
                y_pred = bool_df[thresh].fillna(False).astype(int))

            # specificity for neg task
            ss = tn / (tn + fp)

    #         # npv for positive case
    #         ss = specificity_score(y_true=1-bool_df['any_sbo_surg_enc'],
    #                           y_pred = bool_df[thresh].astype(int))
    #         conf_mtx = confusion


            # This is specificity when defined for the positive case???
            # recall = tp / (tp + fn)
            # recall fn on neg case is computing tn / (tn + fp) ?????
            rs = recall_score(y_true=bool_df['any_sbo_surg_enc'],
                              y_pred = bool_df[thresh].fillna(False).astype(int))
            rs = tp / (tp + fn)
            tpr = rs

            fpr = fp / (fp + tn)

            scores += [[fpr, tpr, rs, ppv]]


            df = time_df[time_df['any_sbo_surg_enc']==1].copy().dropna(subset=[thresh])
            #print(thresh, df.shape, df[df[thresh]==0].shape)
            df = df['time_to_event_enc'] - df[thresh]
            times += [[df.mean(), df.sum()]]

            # Filter out surg ppl, only "save time" for discharge pop
            df = time_df[time_df['any_sbo_surg_enc']==1].copy()
            df[thresh] = df[thresh].fillna(df['time_to_event_enc'])
            #print(df.shape, (df[df[thresh]==0].shape))
            #df = time_df.fillna(subset=[thresh])
            df = df['time_to_event_enc'] - df[thresh]
            times_na += [[df.mean(), df.sum()]]

        scores, times, times_na = np.array(scores), np.array(times), np.array(times_na)

        all_scores.append(scores)
        all_times.append(times)
        all_times_na.append(times_na)

    #return scores, times, times_na
    return all_scores, all_times, all_times_na


# In[359]:


disch_scores, disch_times, disch_times_wna = compute_scores(time_df, bool_df)


# In[350]:


disch_times[0]


# In[327]:


def manual_auc(fpr, tpr):
    fpr = np.concatenate([np.array([1]), fpr, np.array([0])])
    tpr = np.concatenate([np.array([1]), tpr, np.array([0])])
    fpr_series = pd.Series(fpr[::-1])
    tpr = tpr[::-1]
    fpr_diff = fpr_series.diff().fillna(fpr_series).values
    auc = np.dot(fpr_diff, tpr)
    return auc



# In[ ]:





# In[360]:



fig, axes = plt.subplots(1,2, figsize=(15,5))

for i in range(4):
    print('AUC = {}'.format(manual_auc(disch_scores[i][:,0], disch_scores[i][:,1])))

    #plt.figure(figsize=(20,5))
    #plt.subplot(131)

    axes[0].step(disch_scores[i][:,0], disch_scores[i][:,1])
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')

    thresh_num=50
    print('Threshold = {}, Specificity = {}, PPV = {}'.format(thresh_list[thresh_num], disch_scores[i][thresh_num,0], disch_scores[i][thresh_num,3]))
    axes[0].scatter(disch_scores[i][thresh_num,0], disch_scores[i][thresh_num,1], c='red')
    axes[0].grid(True)



    # plt.subplot(132)
    # plt.step(disch_scores[i][:,1], disch_scores[i][:,3])
    # plt.scatter(disch_scores[i][thresh_num,1], disch_scores[i][thresh_num,3], c='red')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.grid(True)

    #axes[1].subplot(133)
    axes[1].step(disch_times_wna[i][:,0], disch_scores[i][:,3], label='axis1',)
    #plt.plot(thresh_list, disch_times_wna[:,0], label='axis1')
    #axes[1].yticks()
    axes[1].set_xlabel('Expected time saved per patient')
    axes[1].set_ylabel('Positive Predictive Value', )
    axes[1].grid(True)
    #ax = plt.gca().twinx()
    #ax.step(thresh_list, disch_scores[:,2], c='tab:orange', )
    #marker='+', s=100
    #ax.set_ylabel('Proportion of ', color='tab:orange')


plt.tight_layout();


# In[ ]:





# In[196]:


def disch_proc(df):
    df = df.copy()
    df.set_index('any_sbo_surg_enc')
    #thresh = np.array([0,0.5])
    thresh_df = pd.concat(map(lambda thresh: (df['pred_ema6'] > thresh).any(), [0, 0.5]), axis=1)
    #thresh_df = (df['pred_ema6'] > thresh)
    #df = pd.concat([df, thresh_df], axis=1)
    #df['disch_ever'] = df.groupby(level=1).transform(lambda x: x.any())
    display(df)
    #df2 = df.groupby(level=1).head(1)
    #return precision_score(y_true=1-df2['any_sbo_surg_enc'], y_pred=df2['disch_ever'])


# In[197]:


y_cp.head(100)[lambda df: df['hsa_cut'] < 121].groupby('hsa_cut').apply(disch_proc)
