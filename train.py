"""
Train models.
"""

import pandas as pd
#import matplotlib.pyplot as plt
from preprocessing import preprocess
from time import time
import argparse

# sklearn
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier

proc_path = 'data/processed/'

parser = argparse.ArgumentParser(description="Specify data and modelling parameters.")
parser.add_argument('--enc', action='store_true',
                    help='Classify on the level of encounters.')
parser.add_argument('--droprates', action='store_true',
                    help='Drop rate columns.')
args = parser.parse_args()
enc = args.enc
drop_rates = args.droprates

if enc:
    print('Just modelling beginning of encounter')
if drop_rates:
    print('Dropping rate columns')

# TODO: Remove fillna here
train = pd.read_pickle(proc_path + 'train.pickle').fillna(0)
val   = pd.read_pickle(proc_path + 'val.pickle').fillna(0)

print(train.any_sbo_surg_enc.value_counts())

print(train.shape)
x_train, y_train, x_train_cols = preprocess(train, enc, drop_rates)
x_val,   y_val,   x_val_cols   = preprocess(val, enc, drop_rates)
print(x_train.shape)

print('Fitting Random Forest...')
start = time()
rf = RandomForestClassifier(class_weight='balanced', n_estimators=40)
rf.fit(x_train, y_train)
print(f'Fit finished in {round(time()-start)}')

# Get metrics
y_pred = rf.predict(x_val)
y_proba = rf.predict_proba(x_val)[:,1]
'''
print('Confusion Matrix:')
print(confusion_matrix(y_val, y_pred))
'''

print(f'AUC ROC score: {round(roc_auc_score(y_val, y_proba), 4)}')
print(f'AUPR score: {round(average_precision_score(y_val, y_proba), 4)}')

'''
def plot_auc(y_val, y_proba):
    fig, axes = plt.subplots(1,2, figsize=(14,6))

    fpr,       tpr,    thresholds = roc_curve(y_val, y_proba)
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

    axes[0].scatter(fpr, tpr)
    axes[0].set(title='ROC Curve',
                xlabel = 'False Positive Rate',
                ylabel= 'True Positive Rate')

    axes[1].scatter(recall, precision)
    axes[1].set(title='Precision-Recall Curve',
                xlabel = 'Recall',
                ylabel= 'Precision')

    plt.savefig('reports/figures/rf.png')

plot_auc(y_val, y_proba)

plt.show()
'''