# If additional attribute transformations desired, the natural step in the workflow
#  is after the attributes have been acquired, and before running fs_proc_algo.py 

import argparse
import yaml
import pandas as pd
from pathlib import Path
import fs_algo.fs_algo_train_eval as fsate
import ast
from collections.abc import Iterable

from typing import Callable
import itertools
import numpy as np
import dask.dataframe as dd
from datetime import datetime
import os
from collections import ChainMap

home_dir = Path.home()
path_tfrm_cfig = Path(f'{home_dir}/git/formulation-selector/scripts/eval_ingest/xssa/xssa_attrs_tform.yaml') 

with open(path_tfrm_cfig, 'r') as file:
    tfrm_cfg = yaml.safe_load(file)

# Read from transform config file:
catgs_attrs_sel = [x for x in list(itertools.chain(*tfrm_cfg)) if x is not None]
idx_tfrm_attrs = catgs_attrs_sel.index('transform_attrs')
idx_file_io = catgs_attrs_sel.index('file_io')
fio = dict(ChainMap(*tfrm_cfg[idx_file_io]['file_io'])) # dict of file input/output, read-only combined view

# Extract desired content from attribute config file
path_attr_config=fsate.build_cfig_path(path_tfrm_cfig, Path(fio.get('name_attr_config')))
attr_cfig = fsate.AttrConfigAndVars(path_attr_config) # TODO consider fsate
attr_cfig._read_attr_config()
# Define all directory paths in case used in f-string evaluation
dir_base = attr_cfig.attrs_cfg_dict.get('dir_base') 
dir_db_attrs = attr_cfig.attrs_cfg_dict.get('dir_db_attrs')
dir_std_base = attr_cfig.attrs_cfg_dict.get('dir_std_base')
datasets = attr_cfig.attrs_cfg_dict.get('datasets')

#%% READ COMIDS FROM CUSTOM FILE (IF path_comids present in tfrm config)
# Extract location of custom file containing comids:
path_comid = eval(f"f'{fio.get('path_comids', None)}'")
ls_comid = list()
# Read in comid from custom file (e.g. predictions)
if path_comid:
    path_comid = Path(path_comid)
    colname_comid = fio.get('colname_comid') # TODO adjust this to fio 
    df_comids = read_df_ext(path_comid)
    ls_comid = ls_comid + df_comids[colname_comid].to_list()

#%%  READ COMIDS GENERATED FROM proc_attr_hydfab 
likely_ds_types = ['training','prediction']
loc_id_col = 'comid'
name_attr_config = fio.get('name_attr_config', None)# TODO read this from the tfrm_attrs config fio 

ls_comids_attrs = list()
if  name_attr_config: 
    # Attribute metadata containing a comid column as standard format 
    path_attr_config = fsate.build_cfig_path(path_tfrm_cfig, name_attr_config)#fsate.build_cfig_path(path_algo_config, name_attr_config)
    ls_comids_attrs = _get_comids_std_attrs(path_attr_config)
    
# Compile unique comid values
comids = list(set(ls_comid + ls_comids_attrs))


# ----------- existing dataset checker ----------- #
# ls_chck <- proc.attr.hydfab::proc_attr_exst_wrap(comid,dir_db_attrs,
#                                                 vars_ls,bucket_conn=NA)

# def proc_attr_exst_wrap(comid, dir_db_attrs, vars_ls, bucket_conn=None):
#     """ Existing attribute data checker.

#     Retrieves the attribute data that already exists in a data storage path for a given `comid`
#     and identifies missing attributes.

#     :param comid: The common identifier USGS location code for a surface water feature.
#     :type comid: str
#     :param dir_db_attrs: Path to the attribute file data storage location.
#     :type dir_db_attrs: str
#     :param vars_ls: Dictionary of variable names grouped by data source.
#     :type vars_ls: dict
#     :param bucket_conn: Cloud connection details if data is stored in S3 or similar (default is None).
#     :type bucket_conn: object, optional

#     :return: Dictionary containing:
#              - `dt_all`: a DataFrame of existing comid data.
#              - `need_vars`: a dictionary containing lists of variable names that need to be downloaded.
#     :rtype: dict

#     :seealso:: `proc.attr.hydfab::proc_attr_exst_wrap`
#     """
#     # Convert dir_db_attrs to a Path object
#     dir_db_attrs = Path(dir_db_attrs)

#     # # Ensure directory exists if not using cloud storage
#     # if not dir_db_attrs.parent.is_dir() and bucket_conn is None:
#     #     dir_db_attrs.parent.mkdir(parents=True, exist_ok=True)

#     if dir_db_attrs.exists():
#             # Load existing dataset if present
#             dataset = pd.read_parquet(dir_db_attrs)
#             dt_all = pd.DataFrame(dataset.to_table().to_pandas())
            
#             need_vars = {}
#             for var_srce, attrs_reqd in vars_ls.items():
#                 # Identify missing attributes
#                 attrs_needed = [attr for attr in attrs_reqd if attr not in dt_all['attribute'].values]
                
#                 if attrs_needed:
#                     need_vars[var_srce] = attrs_needed
#         else:
#             # No subset of variables is present; fetch all for this comid
#             need_vars = vars_ls
#             dt_all = pd.DataFrame()  # Placeholder DataFrame

#         return {'dt_all': dt_all, 'need_vars': need_vars}

    
#TODO name new transformation data as comid_{comid}_tformattrs.parquet in the same directory as the other comid_{comid}_attrs.parquet
#%%

tfrm_cfg_attrs = tfrm_cfg[idx_tfrm_attrs]


proc_tfrm_cfg(tfrm_cfg= tfrm_cfg, idx_tfrm_attrs: int,
                   all_attr_ddf=all_attr_ddf))





# TODO Checkto see if data exist from comid_{comid}_tformattrs.parquet before transforming and writing\
comid = '22152435'
for comid in comids:
#%%
    # Filepath substring structures based on comids 
    fp_struct_std=f'_{comid}_attrs' # The unique string in the filepath name based on standard attributes acquired from external sources
    fp_struct_tfrm=f'_{comid}_tfrmattr' # The unique string in the filepath name based on custom attributes created by RaFTS users


    # Lazy load dask df of transform attributes for a given comid
    tfrm_attr_ddf =  _subset_ddf_parquet_by_comid(dir_db_attrs=dir_db_attrs,
                                                fp_struct=fp_struct_tfrm)
    # TODO define which transformation variables needed

    # TODO loop over tform_type and retr_vars for all possibilities defined in the config file
    
    #%% PARSING THE TRANSFORMATION CONFIG FILE 
    # Create the custom functions
    dict_cstm_vars_funcs = _retr_cstm_funcs(tfrm_cfg_attrs)
    # Note that this is a flattened length size, based on the total # of transformation functions & which transformations are needed
    
    # Desired custom variable names (corresponds to 'attribute' column) 
    dict_all_cstm_vars = dict_cstm_vars_funcs.get('dict_all_cstm_vars')

    # functions: The list of the actual function objects
    dict_func_objs = dict_cstm_vars_funcs['dict_tfrm_func_objs']
    # functions: Desired transformation functions w/ vars (as str objs (corresponds to 'data_source' column))
    dict_all_cstm_funcs = dict_cstm_vars_funcs.get('dict_cstm_func')
    ls_all_cstm_funcs = list(dict_all_cstm_funcs.values())
    # functions: The just-function in string format
    dict_cstm_func = dict_cstm_vars_funcs['dict_tfrm_func']
    # vars: The dict of attributes to aggregate for each custom variable name
    dict_retr_vars = dict_cstm_vars_funcs.get('dict_retr_vars')
    
    #%%  MAYBE DELETE THIS
    # if not tfrm_attr_ddf: # Cre
    #     # TODO perform full attribute acquisition
    #     print("none of the custom attributes exist.")
        

    # else: # Determine which function transformations already exist

    #     # TODO

    #%% IDENTIFY NEEDED ATTRIBUTES/FUNCTIONS
    # ALL attributes for a given comid, read using a file
    all_attr_ddf = _subset_ddf_parquet_by_comid(dir_db_attrs,
                                    fp_struct=comid)

    # Identify the needed functions based on querying the comid's attr data's 'data_source' column
    #  Note the custom attributes used the function string as the 'data_source'
    dict_need_vars_funcs =_id_need_tfrm_attrs(
                            all_attr_ddf=all_attr_ddf,
                            ls_all_cstm_vars=None,
                            ls_all_cstm_funcs = ls_all_cstm_funcs)

    # TODO Check whether all variables used for aggregation exist in parquet files
    # Find the custom variable names we need to create; also the key values in the dicts returned by _retr_cstm_funcs()
    cstm_vars_need =  [k for k, val in dict_all_cstm_funcs.items() if val in dict_need_vars_funcs.get('funcs')]

    #%% Loop over each needed attribute:
    ls_df_rows = list()
    for new_var in cstm_vars_need: 
        if len(cstm_vars_need) != len(dict_need_vars_funcs.get('funcs')):
            raise ValueError("DO NOT PROCEED! Double check assumptions around idx_need indexing")
        
        # Retrieve the transformation function object
        func_tfrm = dict_func_objs[new_var]

        # The attributes used for creating the new variable
        attrs_retr_sub = dict_retr_vars.get(new_var)
        
        # Retrieve the variables of interest for the function
        df_attr_sub = fsate.fs_read_attr_comid(dir_db_attrs, comids_resp=[comid], attrs_sel=attrs_retr_sub,
                        _s3 = None,storage_options=None,read_type='filename')
        
        # Apply transformation
        # Subset data to variables and compute new attribute
        attr_val = _sub_tform_attr_ddf(all_attr_ddf=all_attr_ddf, 
                    retr_vars=attrs_retr_sub, func = func_tfrm)
        
        # Populate new values in the new dataframe
        new_df = _gen_tform_df(all_attr_ddf=all_attr_ddf, 
                            new_var_id=new_var,
                            attr_val=attr_val,
                            tform_type = dict_cstm_func.get(new_var),
                            retr_vars = attrs_retr_sub)
        ls_df_rows.append(new_df)

   

    df_new_vars = pd.concat(ls_df_rows)


    # Create the expected transformation data filepath path
    path_tfrm_comid = _std_attr_filepath(dir_db_attrs=dir_db_attrs,
                       comid=comid,
                       attrtype = 'tfrmattr')
    if path_tfrm_comid.exists():
        df_exst_vars_tfrm = pd
    else:
        df_new_vars.to_parquet(path_tfrm_comid)

    if not df_exst_tfrmattr: # no data exist, write the new df
        # TODO write data
        df_new_vars.to_parquet()
    else:
        # TODO Load existing data, add to it, then write update


    # Load existing attribute filename:
    df_attr_sub = fsate.fs_read_attr_comid(dir_db_attrs, comids_resp=[comid], attrs_sel='all',
                        _s3 = None,storage_options=None,read_type='filename')


    for item in tfrm_cfg[idx_tfrm_attrs]['transform_attrs']:
        for key, value in item.items():
            ls_tfrm_keys = list(itertools.chain(*[[*x.keys()] for x in value]))
            idx_tfrm_type = ls_tfrm_keys.index('tform_type')
            idx_var_desc = ls_tfrm_keys.index('var_desc')
            idx_vars = ls_tfrm_keys.index('vars')
            print(f"Transform Name: {key}")
            tfrm_types = value[idx_tfrm_type]['tform_type']
            print(f"Description: {value[idx_var_desc]['var_desc']}")
            retr_vars = value[idx_vars]['vars']

            # TODO Check to see if attribute already exists, if so read here and skip the rest below

            # Perform aggregation

            for tform_type in tfrm_types:
                # Create name of new attribute
                new_var_id = key.format(tform_type=tform_type)
                print(f"Creating {new_var_id}")





        # TODO change _gen_tform_df to operate on a df rather than ddf
        _gen_tform_df(all_attr_ddf: dd.DataFrame, new_var_id: str,
                    attr_val:float, tform_type: str,
                    retr_vars: str | Iterable)

        attr_vals = df_attr_sub['value'].values()
            # # Retrieve needed attributes for the comid:
            # matching_files = [file for file in Path(dir_db_attrs).iterdir() if file.is_file() and any(sub in file.name for sub in comids)]

        dict_need_vars_funcs 


        # TODO check fp_struct with _attr and w/o _attr once _tformattr written
        # Retrieve the variables for a given location (a dask data.frame)
        all_attr_ddf = _subset_ddf_parquet_by_comid(dir_db_attrs=dir_db_attrs,
                                                    fp_struct=fp_struct_std)
        
        # Identify which custom attributes haven't been created for a location
        ls_need_vars =_id_need_tfrm_attrs(all_attr_ddf, 
                            ls_all_cstm_funcs = ls_all_cstm_funcs)

        # Lazy load all attributes needed for achieving transformation objects
        sub_attr_need_ddf = 

        # TODO enable read/write to file

        # TODO consider creating a tfrm_cfg parser`
tfrm_cfg_attrs = tfrm_cfg[idx_tfrm_attrs]


# TODO identify the names of the desired variables, find which ones don't exist, then only perform transformation and writing if the custom attribute doesn't already exist in the data

# Find which variables have already been created:
subattr_ddf = all_attr_ddf[all_attr_ddf['attribute'].isin(ls_all_cstm_vars)]
subattrs_avail = subattr_ddf['attribute'].unique().collect() # The attributes already present

# Search which custom datasources (aka the function and variables) match
subfunc_ddf = all_attr_ddf[all_attr_ddf['data_source'].isin(ls_all_cstm_funcs)]
subfuncs_avail = subfunc_ddf['attribute'].unique().collect()