"""
Analysis functions for attribute selection

# Usage example
fs_proc_algo.py "/path/to/formulation-selector/scripts/eval_ingest/ealstm/ealstm_train_attrs_31.csv"


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fs_algo.fs_algo_train_eval import AlgoTrainEval, AlgoEvalPlot
import matplotlib
import seaborn as sns
import os

alg_eval_plot = AlgoEvalPlot(train_eval)
train_eval = fsate.AlgoTrainEval(df=df_pred_resp,
                            attrs=attrs_sel,
                            algo_config=algo_config,
                            dir_out_alg_ds=dir_out_alg_ds, dataset_id=ds,
                            metr=metr,test_size=test_size, rs = seed,
                            verbose=verbose)
train_eval.split_data() # Train, test, eval wrapper
df_X, y = train_eval.all_X_all_y()
# TODO remove the above placeholder, just need df_X
#%% 
# TODO define X, metr, dataset plots path, 

# We really only need df_X as the input
# Retrieve the full dataset for assessment

df_corr = df_X.corr()

#%% CORRELATION ANALYSIS: GUIDE USER TO SIMPLIFY TRAINING DATA

def plot_corr_mat(df_X, title='Feature Correlation Matrix') -> matplotlib.figure.Figure:
# TODO EVALUATE EACH DATASET FOR EACH METRIC. Some metrics may be easier to predict than others??
# Calculate the correlation matrix
    df_corr = df_X.corr()

    #  Plot the correlation matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(df_corr, annot=True, cmap ='coolwarm',linewidths=0.5, fmt='.2f')
    plt.title(title)

    fig = plt.gcf()
    return fig

def std_analysis_dir():
    home_dir = str(Path.home())
    dir_anlys_base = Path(f"{home_dir}/noaa/regionalization/data/output/analysis/")
    dir_anlys_base.mkdir(parents=True, exist_ok=True)
    return dir_anlys_base

ds = 'ealstm_test'
def std_corr_path(dir_anlys_base, ds, metr):
    # TODO generate a file of the correlated attributes:

    path_corr_attrs = Path(f"{dir_anlys_base}/{ds}/correlated_attrs_{ds}_{metr}.csv")
    path_corr_attrs.parent.mkdir(parents=True,exist_ok=True)
    return path_corr_attrs

def corr_attrs_thr_table(df_X, 
                        corr_thr = 0.8) ->pd.DataFrame:
    """Create a table of correlated attributes exceeding a threshold, with correlation values

    :param df_X: The attribute dataset
    :type df_X: pd.DataFrame
    :param corr_thr: The correlation threshold, between 0 & 1. Absolute values above this should be reduced, defaults to 0.8
    :type corr_thr: float, optional
    :return: The table of attribute pairings whose absolute correlations exceed a threshold
    :rtype: pd.DataFrame
    """
    df_corr = df_X.corr()

    # Select upper triangle of correlation matrix
    upper = df_corr.abs().where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    upper = df_corr.abs().where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    # Find attributes with correlation greater than a certain threshold
    row_idx, col_idx = np.where(df_corr.abs() > corr_thr)
    df_corr_rslt = pd.DataFrame({'attr1': df_corr.columns[row_idx],
                'attr2': df_corr.columns[col_idx],
                'corr' : [df_corr.iat[row, col] for row, col in zip(row_idx, col_idx)]
                })
    # Remove the identical attributes
    df_corr_rslt = df_corr_rslt[df_corr_rslt['attr1']!= df_corr_rslt['attr2']].drop_duplicates()
    return df_corr_rslt

def write_corr_attrs_thr(df_corr_rslt:pd.DataFrame,path_corr_attrs: str | os.PathLike):
    """Wrapper to generate high correlation pairings table and write to file

    :param df_corr_rslt: _description_
    :type df_corr_rslt: pd.DataFrame
    :param path_corr_attrs: csv write path
    :type path_corr_attrs: str | os.PathLike
    """

    df_corr_rslt.to_csv(path_corr_attrs) # INSPECT THIS FILE 
    print(f"Wrote highly correlated attributes to {path_corr_attrs}")
    print("The user may now inspect the correlated attributes and make decisions on which ones to exclude")

def corr_thr_write_table(df_X:pd.DataFrame,path_corr_attrs:str|os.PathLike,
                       corr_thr=0.8):
    """Wrapper to generate high correlation pairings table above a threshold of interest and write to file
    
    :param df_X: The attribute dataset
    :type df_X: pd.DataFrame
    :param path_corr_attrs: csv write path
    :type path_corr_attrs: str | os.PathLike
    :param corr_thr: The correlation threshold, between 0 & 1. Absolute values above this should be reduced, defaults to 0.8
    :type corr_thr: float, optional
    :return: The table of attribute pairings whose absolute correlations exceed a threshold
    :rtype: pd.DataFrame
    """
    
    df_corr_rslt = corr_attrs_thr_table(df_X,corr_thr)
    write_corr_attrs_thr(df_corr_rslt,path_corr_attrs)
    return df_corr_rslt

# TODO below here
path_corr_attrs = std_corr_path(dir_anlys_base, ds, metr)
path_corr_attrs_fig
title_fig_corr
fig_corr_mat = plot_corr_mat(df_X, title = title_fig_corr)



#%% ATTRIBUTE IMPORTANCE
import fs_algo
def _extr_rf_algo(train_eval:fs_algo.fs_algo_train_eval.AlgoTrainEval):
    
    if 'rf' in train_eval.algs_dict.keys():
        rfr = train_eval.algs_dict['rf']['algo']
    else:
        print("Trained random forest object 'rf' non-existent in the provided AlgoTrainEval class object.",
              "Check to make sure the algo processing config file creates a random forest. Then make sure the ")
        rfr = None
    return rfr

def plot_rf_importance(feat_imprt,attrs, title):
    df_feat_imprt = pd.DataFrame({'attribute': attrs,
                                'importance': feat_imprt}).sort_values(by='importance', ascending=False)
    # Calculate the correlation matrix
    plt.figure(figsize=(10,6))
    plt.barh(df_feat_imprt['attribute'], df_feat_imprt['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Attribute')
    plt.title(title)
    plt.show()

    fig = plt.gcf()
    return fig

def save_feat_imp_fig(fig_feat_imp, path_fig_imp):
    fig_feat_imp.save(path_fig_imp)
    print(f"Wrote feature importance figure to {path_fig_imp}")

rfr = _extr_rf_algo(train_eval)
if rfr:
    feat_imprt = rfr.feature_importances_
    title_rf_imp = f"Random Forest feature importance for {metr}"
    fig_feat_imp = plot_rf_importance(feat_imprt, attrs=df_X.columns, title= title_rf_imp)

#%% PRINCIPAL COMPONENT ANALYSIS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Fit using the 'raw' data
pca = PCA()
pca.fit(df_X) # TODO consider fitting X_train instead
cpts = pd.DataFrame(pca.transform(df_X))
x_axis = np.arange(1, pca.n_components_+1)

# Fit using the scaled data
scaler = StandardScaler().fit(df_X)
df_X_scaled = pd.DataFrame(scaler.transform(df_X), index=df_X.index.values, columns=df_X.columns.values)
pca_scaled = PCA()
pca_scaled.fit(df_X_scaled)
cpts_scaled = pd.DataFrame(pca.transform(df_X_scaled))

# matplotlib boilerplate goes here