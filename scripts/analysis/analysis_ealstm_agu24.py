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

def corr_attrs_thr_table(df_X, path_corr_attrs, corr_thr = 0.8):
    """_summary_

    :param df_X: _description_
    :type df_X: _type_
    :param path_corr_attrs: _description_
    :type path_corr_attrs: _type_
    :param corr_thr: _description_, defaults to 0.8
    :type corr_thr: float, optional
    """
    #corr_thr = 0.8 # The correlation threshold. Absolute values above this should be reduced

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

    # TODO create file write function
    df_corr_rslt.to_csv(path_corr_attrs) # INSPECT THIS FILE 
    print(f"Wrote highly correlated attributes to {path_corr_attrs}")
    print("The user may now inspect the correlated attributes and make decisions on which ones to exclude")
#%% ATTRIBUTE IMPORTANCE
rfr = train_eval.algs_dict['rf']['algo']
feat_imprt = rfr.feature_importances_
title_rf_imp = f"Random Forest feature importance for {metr}"
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