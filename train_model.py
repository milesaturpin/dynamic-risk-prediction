from glmnet_python import glmnet
from glmnet_python import glmnetCoef
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

import sys
sys.path.insert(0, "~/PACE_Home_Drive/sbo-clone4/sbo")
from utils import generate_prefix_dict, generate_suffix_dict, generate_midfix_dict
from preprocessing_exp_weights import preprocess_exp_weights

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score


def c_index(betas, x, y):
    """
    betas - (m,beta_param_shape)
    x - (n_samp,m), DataFrame
    time_of_event - (n_samp,1)
    event_ind - (n_samp)
    Calculate c-index by comparing X @ beta values, and counting
    correct number of rankings.
    """
    # betas shaped like (m, beta_param_shape)
    beta_param_shape = betas.shape[1:]
    time_of_event = y.iloc[:,0].values.reshape((-1,1))
    event_ind = y.iloc[:,1].values
    observed_event_set   = np.where(event_ind == 1)
    unobserved_event_set = np.where(event_ind == 0)
    correct_ranking_count, num_comparisons = np.zeros(beta_param_shape), 0

    # Ranking by linear combination of betas w/ X
    # hazard_pred is (n,m)x(m,beta_param_shape)=(n,beta_param_shape)
    hazard_pred = x.values @ betas
    # (num_obs_events, beta_param_shape)
    obs_hazard_preds = hazard_pred[observed_event_set]
    observed_event_times = time_of_event[observed_event_set]

    for obs_haz_pred, event_time in zip(obs_hazard_preds, observed_event_times):
        # Shape of obs_haz_pred is (beta_param_shape)
        risk_idx   = np.where(time_of_event > event_time)[0]
        risk_preds = hazard_pred[risk_idx]
        # Want shapes: (1, beta_param_shape) > (num_risk, beta_param_shape)
        correct_ranking_count = correct_ranking_count + np.sum(
            obs_haz_pred > risk_preds, axis=0)
        num_comparisons += risk_idx.shape[0]
        # Ties
        tie_idx   = np.where(time_of_event == event_time)[0]
        correct_ranking_count = correct_ranking_count + (tie_idx.shape[0]-1)*0.5
        num_comparisons += tie_idx.shape[0]-1
    # c_index_arr has shape (beta_param_shape,)
    c_index_arr = correct_ranking_count / num_comparisons
    return c_index_arr


def c_index_vs_lambda(fit, x, y):
    log_lambda = np.flip(np.log(fit['lambdau']))
    c_index_arr = c_index(betas=np.flip(glmnetCoef(fit), axis=1), x=x, y=y)
    c_index_df = pd.Series(
        c_index_arr,
        index=log_lambda,
        name='c_index')
    return c_index_df


def fill_na(df, train_means):
    """
    PROTOCOL
    Curr - ffill then fill with training set means
    EMA - mean (should only be at beginning of encounter)
    EMS - 0
    """
    df = df.copy()
    pref_dict = generate_prefix_dict(df)
    df.loc[:,pref_dict['curr']] = (
        df[pref_dict['curr']]
        .fillna(method='ffill')
        .fillna(train_means.loc[pref_dict['curr']]))
    df.loc[:,pref_dict['ema']] = (
        df[pref_dict['ema']]
        .fillna(train_means.loc[pref_dict['ema']]))
    #df['word_log_ratio_img'] = df['word_log_ratio_img'].fillna(train_means.loc['word_log_ratio_img'])
    return df


def scale(train, test):
    train, test = train.copy(), test.copy()
    pref_dict = generate_prefix_dict(train)
    suff_dict = generate_suffix_dict(train)

    #curr_ios  = list(set(pref_dict['curr']) &
    #                (set(suff_dict['io']) | set(suff_dict['occ'])))
    #curr_nums = list(set(pref_dict['curr']) &
    #                (set(suff_dict['vitals']) | set(suff_dict['labs'])))
    #stand_cols  =  pref_dict['ema'] + ['age_enc'] + curr_nums
    #minmax_cols = pref_dict['tsl'] + curr_ios + pref_dict['ems']

    curr_ios  = list(set(pref_dict['curr']) &
                    (set(suff_dict['io']) | set(suff_dict['occ'])))
    curr_nums = list(set(pref_dict['curr']) &
                    (set(suff_dict['vitals']) | set(suff_dict['labs'])))
    stand_cols  =  (pref_dict['ema'] + ['age_enc'] + curr_nums
                    #+ ['word_log_ratio_img']
                   )
    minmax_cols =  curr_ios + ['time_of_day_enc'] + ['hsa_enc']
    robust_cols = pref_dict['tsl'] + pref_dict['ems']

    scaler = StandardScaler()
    train.loc[:,stand_cols] = scaler.fit_transform(train.loc[:,stand_cols])
    test.loc[:,stand_cols]  = scaler.transform(test.loc[:,stand_cols])

    minmax = MinMaxScaler()
    train.loc[:,minmax_cols] = 2 * minmax.fit_transform(train.loc[:,minmax_cols]) - 1
    test.loc[:,minmax_cols]  = 2 * minmax.transform(test.loc[:,minmax_cols]) - 1

    robust = RobustScaler()
    train.loc[:,robust_cols] = 2 * robust.fit_transform(train.loc[:,robust_cols]) - 1
    test.loc[:,robust_cols]  = 2 * robust.transform(test.loc[:,robust_cols]) - 1
    return train, test


def inner_cv_cox(x, y, lambdas, alphas, fold_dict):
    """
    x - pd.DataFrame
    y - pd.DataFrame

    Do inner loop of nested CV. For each fold, fit models on every
    combination of lambda and alpha (elastic net penalty).
    Fit (num_folds * num_lambdas * num_alphas) total models.

    Returns
    inner_perf_arr - a (num_folds, num_lambdas, num_alphas) array
    inner_betas_arr - a (num_folds, num_lambdas, num_alphas, # features) array
    """
    num_folds = len(fold_dict)
    inner_perf_arr = np.zeros((num_folds, len(lambdas), len(alphas)))
    inner_betas_arr = np.zeros((num_folds, len(lambdas), len(alphas), x.shape[1]))

    # Apply preprocessing
    train_means_dict, scalers_dict = cache_preprocessing_info(x, fold_dict)
    fold_generator = generate_fold_datasets(
        x, y, fold_dict, train_means_dict, scalers_dict)

    for i, fold_split in enumerate(fold_generator):
        x_train, x_test, y_train, y_test = fold_split
        start = time.time()
        for j, alpha in enumerate(alphas):
            fit = glmnet(
                x=x_train.values.astype(np.float64),
                y=y_train.values.astype(np.float64),
                family='cox',
                alpha=alpha,
                lambdau=lambdas,
                intr=False)
            betas_arr = glmnetCoef(fit)
            perf_df = c_index_vs_lambda(fit=fit, x=x_test, y=y_test)
            inner_perf_arr[i,:,j] = perf_df.values
            inner_betas_arr[i,:,j,:] = betas_arr.T
        print('Fit finished in {}s'.format(round(time.time()-start, 1)))
    return inner_perf_arr, inner_betas_arr


def auc_vs_lambda(fit, x, y):
    # Note: reversed direction so that lambda goes lo -> hi
    preds = x @ np.flip(glmnetCoef(fit), axis=1)
    calc_auc = lambda s: roc_auc_score(y_true=y.iloc[:,1],
                                       y_score=s.values)
    df = preds.apply(calc_auc).rename('auc').to_frame()
    df['log_lambda'] = np.flip(np.log(fit['lambdau']))
    return df.set_index('log_lambda').auc


def plot_cv_results(inner_perf_arr, lambdas, alphas):
    """Plot AUC vs. Lambda values across runs for diff alphas."""
    grid_shape = int(np.ceil(num_alphas/2)), 2
    fig, axes = plt.subplots(
        *grid_shape, figsize=(8, 1.5*num_alphas), sharey=True, sharex=True)
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
    g.set(xticklabels=np.round(alphas, 2));


def get_best_hparams(perf_arr, lambdas, alphas):
    perf_mean = perf_arr.mean(axis=0)
    max_idx = np.unravel_index(perf_mean.argmax(), perf_mean.shape)
    best_lambda = lambdas[max_idx[0]]
    best_alpha  = alphas[max_idx[1]]
    return best_lambda, best_alpha


def outer_cv_cox(x, y, lambdau, alpha, fold_dict, fold_generator):
    num_folds = len(fold_dict)
    outer_perf_arr = np.zeros(num_folds)
    betas_arr = np.zeros((num_folds, x.shape[1]))

    for i, fold_split in enumerate(fold_generator):
        x_train, x_test, y_train, y_test = fold_split
        start = time.time()

        fit = glmnet(
            x=x_train.values.astype(np.float64),
            y=y_train.values.astype(np.float64),
            family='cox',
            alpha=alpha,
            # Note: supposed to be array
            # Note: included 0.0129 because of weird quirk
            lambdau=np.array([lambdau,0.0129]),
            intr=False)
        mid = time.time()
        print('Fit finished in {}s'.format(round(mid-start, 1)))
        betas = glmnetCoef(fit, s=np.array([lambdau]))
        betas_arr[i,:] = betas.flatten()
        preds = x_test @ betas
        #auc = roc_auc_score(y_true=y_test.iloc[:,1], y_score=preds.values)
        perf = c_index(betas, x_test, y_test)
        outer_perf_arr[i] = perf
    return betas_arr, outer_perf_arr


def permutation_importance(x, y, betas_arr, perm_group_dict):
    """
    betas_arr - (num_folds, p)
    NOTE: dividing by number of elts in the group for avg feat imp
    """
    perm_imp_arr = np.zeros((1, len(perm_group_dict.keys())))
    # Shape of perf is (num_folds)
    perf = c_index(betas_arr, x, y)
    for i, (group, group_columns) in enumerate(perm_group_dict.items()):
        x_perm = x.copy()
        x_perm.loc[:, group_columns] = x_perm.loc[:, group_columns].apply(
            lambda s: np.random.permutation(s))
        #perf_perm = roc_auc_score(y_true=y_test.iloc[:,1], y_score=preds_perm)
        perf_perm = c_index(betas_arr, x_perm, y)
        # NOTE: dividing by group size
        perm_imp_arr[0,i]= (perf - perf_perm) / len(group_columns)
    return perm_imp_arr


def permutation_importance_cv(x, y, betas_arr, fold_dict, fold_generator, perm_group_dict=None):
    if perm_group_dict is None:
        perm_group_dict = {col:[col] for col in x.columns}

    num_folds = len(fold_dict)
    perm_imp_arr = np.zeros((num_folds, len(perm_group_dict.keys())))

    for i, fold_split in enumerate(fold_generator):
        x_train, x_test, y_train, y_test = fold_split
        start=time.time()
        betas = betas_arr[i,:]
        perm_imp_arr[i,:] = permutation_importance(
            x_test, y_test, betas, perm_group_dict)
        print('Permutation importance finished in {}s'.format(round(time.time()-start, 1)))

    perm_imp_df = pd.DataFrame(perm_imp_arr, columns=perm_group_dict.keys())
    return perm_imp_df


def generate_fold_dict(sbo, num_folds=10, epsilon=0.03):
    full_prop = (
        sbo.reset_index()
        [['mrn', 'id', 'any_sbo_surg_enc']]
        .drop_duplicates()
        [['id', 'any_sbo_surg_enc']]
        .any_sbo_surg_enc
        .value_counts()
        .transform(lambda x: x/x.sum())
        .values)
    mrns = sbo.reset_index().mrn.drop_duplicates()
    seed = 0
    fold_dict = generate_fold_helper(
        sbo, mrns, num_folds-1, epsilon, seed, full_prop)
    return fold_dict

def generate_fold_helper(sbo, mrns, fold_num, epsilon, seed, full_prop):
    print('Fold: {}'.format(fold_num))
    if fold_num == 0:
        test  = sbo.loc[mrns.values, :]
        prop = (
            test.reset_index()
            [['id', 'any_sbo_surg_enc']]
            .drop_duplicates()
            .any_sbo_surg_enc
            .value_counts(normalize=True)
            .values)
        diff = np.abs(prop - full_prop)[0]
        assert diff < epsilon
        fold_dict = {fold_num: mrns.values}
    else:
        maxdiff = 1
        while maxdiff > epsilon:
            seed += 1
            train_mrn, test_mrn = train_test_split(
                mrns, test_size=1/(fold_num+1), random_state=seed)
            train = sbo.loc[train_mrn.values, :]
            test  = sbo.loc[test_mrn.values, :]
            print('Trying new seed: ' + str(seed))
            diffs = []
            for df in [train, test]:
                prop = (
                    df.reset_index()
                    [['id', 'any_sbo_surg_enc']]
                    .drop_duplicates()
                    .any_sbo_surg_enc
                    .value_counts(normalize=True)
                    .values)
                diff = np.abs(prop - full_prop)[0]
                #print(diff)
                diffs += [diff]
            maxdiff = max(diffs)

        fold_dict = {fold_num: test_mrn.values}
        fold_dict.update(generate_fold_helper(
            train, train_mrn, fold_num-1, epsilon, seed, full_prop))
    return fold_dict

def cache_preprocessing_info(x, fold_dict):
    print('\n>>> Caching preprocessing...')
    suff_dict = generate_suffix_dict(x)
    pref_dict = generate_prefix_dict(x)
    train_means_dict = dict()
    scalers_dict = dict()
    for i, (fold_num, mrns) in enumerate(fold_dict.items()):
        print('Fold {}'.format(i))
        start = time.time()
        x_train = x.copy().drop(mrns)
        # Calculate train means
        x_train.loc[:,suff_dict['img']] = x_train.loc[:,suff_dict['img']].fillna(0)
        train_means = x_train.mean(axis=0)
        x_train = x_train.fillna(value=train_means.to_dict())

        # Fit Scaler objects
        standard = StandardScaler()
        minmax = MinMaxScaler()
        robust = RobustScaler()
        curr_ios  = list(set(pref_dict['curr']) &
                        (set(suff_dict['io']) | set(suff_dict['occ'])))
        curr_nums = list(set(pref_dict['curr']) &
                        (set(suff_dict['vitals']) | set(suff_dict['labs'])))
        stand_cols  =  pref_dict['ema'] + ['age_enc'] + curr_nums
        minmax_cols =  curr_ios + ['hsa_enc']
        #, 'time_of_day_enc', 'duke_loc_enc',
        #                           'past_sbo_enc', 'raleigh_loc_enc', 'regional_loc_enc',
        #                           'hsa_enc'] + suff_dict['img']
        robust_cols = pref_dict['tsl'] + pref_dict['ems']



        standard.fit(x_train[stand_cols])
        minmax.fit(x_train[minmax_cols])
        robust.fit(x_train[robust_cols])

        train_means_dict[i] = train_means
        scalers_dict[i] = [standard, minmax, robust]
        print('Finished in {}s'.format(round(time.time()-start, 1)))
    return train_means_dict, scalers_dict


def generate_fold_datasets(x, y, fold_dict, train_means_dict, scalers_dict):
    """
    Usage:
    fold_dataset_generator = generate_fold_datasets(
        x, y, fold_dict, train_means_dict, scalers_dict)
    for fold_split in fold_dataset_generator:
        x_train, x_test, y_train, y_test = fold_split
        # Do something

    Or if only want one fold:
    fold_dataset_generator = generate_fold_datasets(
        x, y, fold_dict, train_means_dict, scalers_dict)
    x_train, x_test, y_train, y_test = next(fold_dataset_generator)
    # Do something

    """
    print('\n>>> Applying preprocessing...')
    suff_dict = generate_suffix_dict(x)
    pref_dict = generate_prefix_dict(x)
    for i, (fold_num, mrns) in enumerate(fold_dict.items()):
        print('Fold {}'.format(i))
        start = time.time()
        xf, yf = x.copy(), y.copy()
        # Fill NA
        train_means, scalers = train_means_dict[i], scalers_dict[i]
        xf.loc[:,suff_dict['img']] = xf.loc[:,suff_dict['img']].fillna(0)
        xf = xf.fillna(value=train_means.to_dict())
        mid = time.time()
        print('Filled NA in {}s'.format(round(mid-start,1)))

        # Apply scaler objects
        curr_ios  = list(set(pref_dict['curr']) &
                        (set(suff_dict['io']) | set(suff_dict['occ'])))
        curr_nums = list(set(pref_dict['curr']) &
                        (set(suff_dict['vitals']) | set(suff_dict['labs'])))
        stand_cols  = pref_dict['ema'] + ['age_enc'] + curr_nums
        minmax_cols =  curr_ios + ['hsa_enc']
        #, 'time_of_day_enc', 'duke_loc_enc',
        #                           'past_sbo_enc', 'raleigh_loc_enc', 'regional_loc_enc',
        #                           'hsa_enc'] + suff_dict['img']
        robust_cols = pref_dict['tsl'] + pref_dict['ems']

        standard, minmax, robust = scalers
        xf.loc[:,stand_cols] = standard.transform(xf[stand_cols])
        xf.loc[:,minmax_cols] = minmax.transform(xf[minmax_cols])
        xf.loc[:,robust_cols] = robust.transform(xf[robust_cols])

        x_train, y_train = xf.drop(mrns), yf.drop(mrns)
        x_test, y_test = xf.loc[mrns], yf.loc[mrns]
        print('Scaled in {}s'.format(round(time.time()-mid,1)))
        yield x_train, x_test, y_train, y_test


def make_predictions(x, y, betas_arr, fold_dict, fold_generator):
    x, y = x.copy(), y.copy()
    num_folds = len(fold_dict)
    y_train_list = []
    y_test_list = []
    for i, fold_split in enumerate(fold_generator):
        x_train, x_test, y_train, y_test = fold_split
        betas = betas_arr[i,:]
        train_pred = x_train @ betas
        test_pred = x_test @ betas
        y_train['pred'] = train_pred
        y_test['pred']  = test_pred
        y_train['fold'] = i
        y_test['fold'] = i
        y_train_list += [y_train]
        y_test_list += [y_test]
    y_train_pred = pd.concat(y_train_list)
    y_test_pred = pd.concat(y_test_list)
    return y_train_pred, y_test_pred


def main(x, y, run_inner_fold=True, compute_perm_imp=True):
    #444
    np.random.seed(449)
    time1 = time.time()
    x, y = x.copy(), y.copy()

    enc = (
        y.reset_index(level=2, drop=True).reset_index().drop_duplicates()
        .set_index(['mrn', 'id']))

    # TODO: figure out if this is a problem

    y['hsa'] = y.index.get_level_values(2)
    y.loc[y.any_sbo_surg_enc == 0, 'time_to_event_enc'] = (
        y.time_to_event_enc.max() - y.loc[y.any_sbo_surg_enc == 0, 'hsa'] )
    y=y.drop('hsa',1)
    y.loc[y['time_to_event_enc'] == 0, 'time_to_event_enc'] = 0.01

    num_folds, num_lambdas, num_alphas  = 5, 20, 3
    lambdas = np.array([np.exp(x) for x in np.linspace(-4, 6, num_lambdas)])
    #alphas = np.array([x**2 for x in np.linspace(0,1,num_alphas)]).round(3)
    # Don't need to add zero because just round
    alphas = np.array(
        [0.1**x for x in np.linspace(0,4,num_alphas)[::-1]]).round(3)
    #alphas = np.array([0])

    print('\n>>> Generating {} group stratified folds...'.format(num_folds))
    group_fold_dict = generate_fold_dict(enc, num_folds, 0.01)
    inner_group_fold_dict = group_fold_dict.copy()
    del inner_group_fold_dict[num_folds-1]

    x_inner = x.drop(group_fold_dict[num_folds-1])
    y_inner = y.drop(group_fold_dict[num_folds-1])
    time2 = time.time()
    print('Finished in {}s'.format(round(time2-time1,1)))


    if run_inner_fold:
        print('\n>>> Running inner CV with {} folds...'.format(num_folds-1))
        inner_perf_arr, inner_betas_arr = inner_cv_cox(
            x=x_inner,
            y=y_inner,
            lambdas=lambdas,
            alphas=alphas,
            fold_dict=inner_group_fold_dict)
        time3 = time.time()
        print('Fit finished in {}s'.format(round(time3-time2,1)))

        lambda_opt, alpha_opt = get_best_hparams(inner_perf_arr, lambdas, alphas)
    else:
        time3 = time.time()
        inner_perf_arr, inner_betas_arr = None, None
        lambda_opt, alpha_opt = 0.25, 0.0
        #lambda_opt, alpha_opt = 1e-3, 0.0

    print('\n>>> Running outer CV with {} folds, \nlambda* = {}, alpha* = {}...'
          .format(num_folds, lambda_opt,alpha_opt ))
    # Cache preprocessing
    train_means_dict, scalers_dict = cache_preprocessing_info(x, group_fold_dict)
    fold_generator = generate_fold_datasets(
        x, y, group_fold_dict, train_means_dict, scalers_dict)
    betas_arr, perf_arr = outer_cv_cox(
        x, y,
        lambda_opt,
        alpha_opt,
        group_fold_dict,
        fold_generator)
    time4 = time.time()
    print('Fit finished in {}s'.format(round(time4-time3,1)))

    if compute_perm_imp:
        print('\n>>> Computing permutation importance...')
        pref_dict = generate_prefix_dict(x_train)
        suff_dict = generate_suffix_dict(x_train)
        mid_dict = generate_midfix_dict(x_train)
        mid_dict['bp'] = mid_dict['bp_sys'] + mid_dict['bp_dia']
        #perm_group_dict = {col:[col] for col in x_samp_full.columns}
        fold_generator = generate_fold_datasets(
            x, y, group_fold_dict, train_means_dict, scalers_dict)
        perm_group_dict = mid_dict
        perm_imp_df = permutation_importance_cv(
            x, y,
            betas_arr,
            group_fold_dict,
            fold_generator,
            perm_group_dict)

    else:
        perm_imp_df = None
    time5 = time.time()
    print('Fit finished in {}s'.format(round(time5-time4,1)))


    print('\n>>> Make predictions...')
    fold_generator = generate_fold_datasets(
        x, y, group_fold_dict, train_means_dict, scalers_dict)
    y_train_pred, y_test_pred = make_predictions(
        x, y,
        betas_arr,
        group_fold_dict,
        fold_generator)
    time6 = time.time()
    print('Predictions finished in {}s'.format(round(time6-time5,1)))
    """
    epsilon, lambdas, alphas, num_folds

    generate_fold_dict -> group_fold_dict
    inner_cv_cox -> inner_perf_arr, inner_betas_arr
    outer_cv_cox -> betas_arr, perf_arr
    permutation_importance_cv -> perm_imp_df
    make_predictions -> y_train_pred, y_test_pred
    """
    betas_df = pd.DataFrame(betas_arr, columns=x.columns)

    result_list = [
        lambdas,
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
        scalers_dict
    ]
    return result_list

if __name__ == "__main__":
    (sbo_x, sbo_y, _) = preprocess_exp_weights(
        rebuild=False,
        time_to_event=True,
        scale_feat=False,
        fill_null=False,
        ffill=True,
        #custom_tag='sboimg'
        custom_tag='hand'
    )

    result_list = main(
        sbo_x, sbo_y, run_inner_fold=False, compute_perm_imp=False)

    with open('../data/processed/result_list_hand.pickle','wb') as f:
        pickle.dump(result_list, f)