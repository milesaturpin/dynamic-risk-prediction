#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

from utils import generate_prefix_dict, generate_suffix_dict, generate_midfix_dict

InteractiveShell.ast_node_interactivity = "all"


# In[36]:





# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from preprocessing_exp_weights import preprocess_exp_weights

(x, y, x_cols) = preprocess_exp_weights(
    rebuild=False, time_to_event=True, scale_feat=False, fill_null=False, custom_tag='noimg')
#img2

suff_dict = generate_suffix_dict(x)


# In[4]:


#x = x.drop(list(set(suff_dict['img'])-{'ind12_word_log_ratio_img','ind48_word_log_ratio_img'}), 1)

pref_dict = generate_prefix_dict(x)
suff_dict = generate_suffix_dict(x)
mid_dict = generate_midfix_dict(x)
#mid_dict['bp'] = mid_dict['bp_sys'] + mid_dict['bp_dia']


# In[5]:


pref_dict.keys()
suff_dict.keys()
mid_dict.keys()


# In[13]:


def plot_cv_results(inner_perf_arr, lambdas, alphas):
    """Plot AUC vs. Lambda values across runs for diff alphas."""
    num_alphas = alphas.shape[0]
    grid_shape = int(np.ceil(num_alphas/2)), 2
    fig, axes = plt.subplots(
        *grid_shape, figsize=(8, 1.5*num_alphas), sharey=True, sharex=True)
    axes = axes.reshape(grid_shape)
    log_lambdas = np.round(np.log(lambdas), 1)
    for k in range(num_alphas):
        ax_idx = np.unravel_index(indices=k, dims=grid_shape)
        perf_vs_lambda_df = pd.DataFrame(
            inner_perf_arr[:,:,k].T, index=log_lambdas)
        axes[ax_idx].plot(perf_vs_lambda_df.mean(axis=1))
        axes[ax_idx].fill_between(
            perf_vs_lambda_df.index,
            perf_vs_lambda_df.min(axis=1),
            perf_vs_lambda_df.max(axis=1),
            alpha=0.1)
        axes[ax_idx].set(title='alpha={}'.format(alphas[k]),
        xlabel='lambdas', ylabel='c-index')
        #axes[ax_idx].set_xticks(log_lambdas)
    plt.tight_layout()
    # Heatmap
    plt.figure(figsize=(10,8))
    g = sns.heatmap(np.mean(inner_perf_arr, axis=0))
    g.set(yticklabels=np.round(lambdas, 5))
    for tick in g.get_yticklabels():
        tick.set_rotation(0)
    g.set(xticklabels=np.round(alphas, 4))


# In[4]:


#img_full?
with open('data/processed/result_list_noimg.pickle','rb') as f:
    result_list = pickle.load(f)

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


# In[5]:


x


# In[5]:


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


# In[6]:


y_test_ema = pd.concat([y_test_pred.drop('pred',1), test_pred_ema], 1)
y_test_ema = y_test_ema.groupby(level=[0,1]).apply(fill_in_y)


# In[7]:


hsa_bins = np.arange(-1,504,12)
y_test_ema['hsa_cut'] = pd.cut(y_test_ema.hsa, bins=hsa_bins).apply(lambda x: x.right+1).astype(int)
tte_bins = np.arange(-1,504,12)
y_test_ema['tte_cut'] = pd.cut(y_test_ema.time_to_event_enc, bins=tte_bins).apply(lambda x: x.right).astype(int)


# In[8]:


compute_target = lambda df, win: (df.any_sbo_surg_enc.astype(bool)
                                  & (df.time_to_event_enc < win)).astype(int)

y_test_ema['target12']  = compute_target(y_test_ema, 12)
y_test_ema['target24']  = compute_target(y_test_ema, 24)
y_test_ema['target48']  = compute_target(y_test_ema, 48)
y_test_ema['target72']  = compute_target(y_test_ema, 72)


# ## Binned Residuals

# In[31]:


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

y_test_ema['prob'] = sigmoid(y_test_ema['pred_ema'])
y_test_ema['prob_bin'] = (
    #pd.cut(y_test_ema['prob'], bins=np.linspace(0,1,100+1))
    pd.qcut(y_test_ema['prob'], q=200)
    .apply(lambda x: x.right).astype(float))
y_test_ema['resid'] = y_test_ema['target24'] - y_test_ema['prob']
temp = y_test_ema.groupby('prob_bin')[['resid','hsa', 'any_sbo_surg_enc']].mean()
#plt.scatter(y_test_ema['prob'],y_test_ema['resid'])
plt.scatter(temp.index, temp['resid'], s=5, c=temp['hsa'])
plt.grid(True)


# In[32]:


temp = y_test_ema.groupby('prob_bin')[['resid', 'hsa','any_sbo_surg_enc']].mean()
plt.scatter(temp.index, temp['any_sbo_surg_enc'], c=temp['hsa'])
plt.grid(True)


# In[33]:


temp = y_test_ema.groupby('hsa_cut')[['resid', 'hsa','any_sbo_surg_enc']].mean()
plt.scatter(temp.index, temp['resid'])
plt.grid(True)


# In[36]:


fig,axes=plt.subplots(1,2,figsize=(13,5))

var = 'curr_bp_sys_vitals'
x_pred['var_cut'] = (
    #pd.cut(y_test_ema['prob'], bins=np.linspace(0,1,100+1))
    pd.qcut(x_pred[var], q=50)
    .apply(lambda x: x.mid).astype(float))
temp = (x_pred.groupby('var_cut')
        [['resid', 'hsa','any_sbo_surg_enc']]
        .mean())
axes[0].scatter(temp.index, temp['resid'],c=temp['hsa'])
axes[0].grid(True)
axes[0].set_title(f'Average residuals vs {var}')

temp = (x_pred.groupby('hsa_cut')
    [var]
    .quantile(np.linspace(0,1,5))
    #.unstack().T
    #.boxplot(sym='', ax=axes[1], whis=[0.05,0.95], vert=False))
       )
plt.scatter(temp.values, temp.index.get_level_values(0))

axes[1].set_title(f'{var} vs HSA')
axes[1].grid(True)
plt.tight_layout()


# In[37]:


var = 'curr_bp_sys_vitals'
x_pred[var].hist(bins=20)


# In[38]:


x_pred['var_cut'].hist()


# In[39]:


y_test_ema['resid'].hist()


# In[ ]:





# ## More

# In[14]:


x_pred = pd.concat([x, y_test_ema], 1, join='inner').loc[:,lambda df: ~df.columns.duplicated()]


# In[15]:


plot_cv_results(inner_perf_arr, lambdas, alphas)


# In[9]:


plt.boxplot(perf_arr);

betas_df.plot(kind='box', figsize=(10,40), vert=False);

# perm_group_dict = mid_dict
# perm_imp_df.plot(
#     kind='box', figsize=(10, max([3, len(perm_group_dict.keys())/4]) ),
#                      vert=False)
plt.tight_layout();


# In[28]:


tte_auc_arr, hsa_auc_arr, tte_bin_labels, hsa_bin_labels = auc_over_time(
    y_test_pred, group_fold_dict,
    tte_q=5, tte_range=24,
    hsa_q=5, hsa_range=(0,128))


# In[30]:


tte_auc_arr, hsa_auc_arr, tte_bin_labels, hsa_bin_labels = auc_over_time(
    y_train_pred, group_fold_dict,
    tte_q=5, tte_range=24,
    hsa_q=5, hsa_range=(0,128))


# In[14]:


fig,axes = plt.subplots(2,1,figsize=(12,9))
for name, group in y_test_pred[lambda df: df.any_sbo_surg_enc == 1].groupby(level=[0,1]):
    axes[0].plot(group['time_to_event_enc'], group['pred'], alpha=0.06, c='black')
    axes[0].set_ylim([-2,3])
for name, group in y_test_pred[lambda df: df.any_sbo_surg_enc == 0].groupby(level=[0,1]):
    axes[1].plot(group['time_to_event_enc'], group['pred'], alpha=0.03,c='black')
    axes[1].set_ylim([-2,3])
plt.tight_layout();


# In[20]:


fig,axes = plt.subplots(1,1,figsize=(12,6))
for name, group in y_test_ema[lambda df: df.any_sbo_surg_enc == 1][lambda df: df.hsa < 200].groupby(level=[0,1]):
    axes.plot(group['hsa'], group['pred_ema6'], alpha=0.1, c='red')
    axes.set_ylim([-1,1.5])
for name, group in y_test_ema.head(100000)[lambda df: df.any_sbo_surg_enc == 0][
                                            lambda df: df.hsa < 200].groupby(level=[0,1]):
    axes.plot(group['hsa'], group['pred_ema6'], alpha=0.1,c='black')

hsa_mod_fn = lambda t: np.exp(-t/100)
ts = np.linspace(0,200,100)
axes.plot(ts, hsa_mod_fn(ts) -0.4 )

plt.tight_layout();


# In[54]:


x_pred.reset_index(level=2, drop=True).any_sbo_surg_enc.reset_index().drop_duplicates()


# In[110]:


max_hour = df_to_plot.reset_index().hour_since_adm.max()
prev_index = df_to_plot.index
(df_to_plot.reindex(np.arange(max_hour + 1))
 ['pred'].ewm(halflife=6, ignore_na=False).mean()
 .reindex(prev_index)
 .plot())


# In[106]:


max_hour = df_to_plot.reset_index().hour_since_adm.max()
(df_to_plot['pred'].dropna().ewm(halflife=6).mean()
 #.reindex(np.arange(max_hour + 1)).fillna(method='ffill')
 .plot())


# In[61]:


# 'AA8915',114609870
df_to_plot = x_pred.loc['AP3120',126874253]
num_plots = (53+1+2)*2
grid_shape = num_plots//2+1,2
fig,axes= plt.subplots(*grid_shape, figsize=(15,80))
axes = axes.reshape(grid_shape)
axes[0,0].plot(df_to_plot['pred'].dropna())
axes[0,0].plot(df_to_plot['pred'].dropna().ewm(halflife=1).mean())
axes[0,0].plot(df_to_plot['pred'].dropna().ewm(halflife=3).mean())
axes[0,0].set_title('pred')
i=1
for name, cols in mid_dict.items():
#for i in range(1,num_plots+1):
    grid_idx = np.unravel_index(i, grid_shape)
    for col in cols[:-1]:
        axes[grid_idx].plot(df_to_plot[col].dropna())
    axes[grid_idx].set_title(name)
    i+=1

for col in pref_dict['tsl']:
#for i in range(1,num_plots+1):
    grid_idx = np.unravel_index(i, grid_shape)
    axes[grid_idx].plot(df_to_plot[col].dropna())
    axes[grid_idx].set_title(col)
    i+=1
plt.tight_layout();


# In[62]:


# 'AA8915',114609870
df_to_plot = x_pred.loc['D1360573',144615046]
num_plots = (56+1+2)*2
grid_shape = num_plots//2+1,2
fig,axes= plt.subplots(*grid_shape, figsize=(15,110))
axes = axes.reshape(grid_shape)
axes[0,0].plot(df_to_plot['pred'].dropna())
axes[0,0].plot(df_to_plot['pred_ema'].dropna())
axes[0,0].set_title('pred')
i=1
for name, cols in mid_dict.items():
#for i in range(1,num_plots+1):
    grid_idx = np.unravel_index(i, grid_shape)
    for col in cols[:-1]:
        axes[grid_idx].plot(df_to_plot[col].dropna())
    axes[grid_idx].set_title(name)
    i+=1

for col in pref_dict['tsl']:
#for i in range(1,num_plots+1):
    grid_idx = np.unravel_index(i, grid_shape)
    axes[grid_idx].plot(df_to_plot[col].dropna())
    axes[grid_idx].set_title(col)
    i+=1
plt.tight_layout();


# In[26]:


(x_pred.groupby(level=[0,1]).filter(lambda df: df.time_to_event_enc.max() < 120).groupby(level=[0,1]).tail(1)
 .sort_values('pred', ascending=False))


# In[ ]:



