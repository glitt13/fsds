import argparse
import yaml
import joblib
import fs_algo.fs_algo_train_eval as fsate
import pandas as pd
from pathlib import Path
import ast
import warnings
import os
import numpy as np

# TODO create a function that's flexible/converts user formatted checks (a la fs_proc)


# Predict values and evaluate predictions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'process the prediction config file')
    parser.add_argument('path_pred_config', type=str, help='Path to the YAML configuration file specific for prediction.')
    # NOTE pred_config should contain the path for path_algo_config
    args = parser.parse_args()

    home_dir = Path.home()
    path_pred_config = Path(args.path_pred_config) #Path(f'{home_dir}/git/formulation-selector/scripts/eval_ingest/xssa/xssa_pred_config.yaml') 
    with open(path_pred_config, 'r') as file:
        pred_cfg = yaml.safe_load(file)

    #%%  READ CONTENTS FROM THE ATTRIBUTE CONFIG
    path_attr_config = fsate.build_cfig_path(path_pred_config,pred_cfg.get('name_attr_config',None))
    attr_cfig = fsate.AttrConfigAndVars(path_attr_config)
    attr_cfig._read_attr_config()

    dir_base = attr_cfig.attrs_cfg_dict.get('dir_base')
    dir_std_base = attr_cfig.attrs_cfg_dict.get('dir_std_base')
    dir_db_attrs = attr_cfig.attrs_cfg_dict.get('dir_db_attrs')
    datasets = attr_cfig.attrs_cfg_dict.get('datasets') # Identify datasets of interest
    attrs_sel = attr_cfig.attrs_cfg_dict.get('attrs_sel', None)

    #%% ESTABLISH ALGORITHM FILE I/O
    dir_out = fsate.fs_save_algo_dir_struct(dir_base).get('dir_out')
    dir_out_alg_base = fsate.fs_save_algo_dir_struct(dir_base).get('dir_out_alg_base')
    #%% PREDICTION FILE'S COMIDS (IMPLICIT ASSUMPTION: Each dataset processes the same IDS)
    path_meta_pred = pred_cfg.get('path_meta')
    comid_pred_col = pred_cfg.get('pred_file_comid_colname')
    write_type = pred_cfg.get('write_type')
    ds_type = pred_cfg.get('ds_type')
    
    # f-string formatting of 
    #path_pred_locs = f'{path_meta_pred}'

    #comids_pred = fsate._read_pred_comid(path_pred_locs, comid_pred_col )

    #%% prediction config
    resp_vars = pred_cfg.get('algo_response_vars')
    algos = pred_cfg.get('algo_type')



    #%% Run prediction
    for ds in datasets:

         # f-string formatting of the attribute metadata's filepath
        path_pred_locs = f'{path_meta_pred}'.format(dir_std_base=dir_std_base,ds=ds,ds_type=ds_type, write_type=write_type)

        comids_pred = fsate._read_pred_comid(path_pred_locs, comid_pred_col )

        #%%  Read in predictor variable data (aka basin attributes) 
        # Read the predictor variable data (basin attributes) generated by proc.attr.hydfab
        df_attr = fsate.fs_read_attr_comid(dir_db_attrs, comids_pred, attrs_sel = attrs_sel,
                                        read_type = 'filename',
                                        _s3 = None,storage_options=None)
        # Convert into wide format for model training
        df_attr_wide = df_attr.pivot(index='featureID', columns = 'attribute', values = 'value')

        # Run predictions & save output
        dir_out_alg_ds = Path(dir_out_alg_base/Path(ds))
        print(f"PREDICTING algorithm for {ds}")
        for metric in resp_vars:
            for algo in algos:
                path_algo = fsate.std_algo_path(dir_out_alg_ds, algo=algo, metric=metric, dataset_id=ds)
                if not Path(path_algo).exists():
                    raise FileNotFoundError(f"The following algorithm path does not exist: \n{path_algo}")


                # Read in the algorithm's pipeline
                pipe = joblib.load(path_algo)
                feat_names = list(pipe.feature_names_in_)
                df_attr_sub = df_attr_wide[feat_names]

                # Perform prediction
                resp_pred = pipe.predict(df_attr_sub)

                # compile prediction results:
                df_pred =pd.DataFrame({'comid':comids_pred,
                             'prediction':resp_pred,
                             'metric':metric,
                             'dataset':ds,
                             'algo':algo,
                             'name_algo':Path(path_algo).name})
                
                path_pred_out = fsate.std_pred_path(dir_out,algo=algo,metric=metric,dataset_id=ds)
                # Write prediction results
                df_pred.to_parquet(path_pred_out)
                print(f"   Completed {algo} prediction of {metric}")