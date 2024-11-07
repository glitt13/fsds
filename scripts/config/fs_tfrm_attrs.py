# If additional attribute transformations desired, the natural step in the workflow
#  is after the attributes have been acquired, and before running the fs_proc_algo.py 

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

home_dir = Path.home()
path_tfrm_cfig = Path(f'{home_dir}/git/formulation-selector/scripts/eval_ingest/xssa/xssa_attrs_tform.yaml') 

with open(path_tfrm_cfig, 'r') as file:
    tfrm_cfg = yaml.safe_load(file)

# Read from transform config file:
catgs_attrs_sel = [x for x in list(itertools.chain(*tfrm_cfg)) if x is not None]
idx_tfrm_attrs = catgs_attrs_sel.index('transform_attrs')
idx_file_io = catgs_attrs_sel.index('file_io')
fio = tfrm_cfg[idx_file_io]['file_io'][idx_file_io]

# Extract desired content from attribute config file
path_attr_config=Path(path_tfrm_cfig.parent/Path(fio.get('name_attr_config')))
attr_cfig = fsate.AttrConfigAndVars(path_attr_config) # TODO consider fsate
attr_cfig._read_attr_config()
dir_db_attrs = attr_cfig.attrs_cfg_dict.get('dir_db_attrs')

# Extract location of file containing comids:
path_comid = Path(fio) #TODO adjust this to fio contents
comid_col = 'comid' # TODO adjust this to fio 

# TODO read in comid from custom file (e.g. predictions)

# TODO read in file for comids. Allow .csv or .parquet format
if 'csv' in path_comid.suffix():
    df_comids = pd.read_csv(path_comid)
elif 'parquet' in path_comid.suffix():
    df_comids = pd.read_parquet(path_comid)
else:
    raise ValueError("Expecting path to file containing comids to be csv or parquet file")

comids = 

# TODO read in comid from standardized dataset (e.g. post-fs_proc)


# TODO define comids and loop (likely place in a wrapper)

# TODO enable read/write to file


#TODO name new transformation data as comid_{comid}_tformattrs.parquet in the same directory as the other comid_{comid}_attrs.parquet

# TODO Checkto see if data exist from comid_{comid}_tformattrs.parquet before transforming and writing
comid = '22152435'

# Filepath substring structures based on comids 
fp_struct_std=f'*_{comid}_attr*' # The unique string in the filepath name based on standard attributes acquired from external sources
fp_struct_tfrm=f'*_{comid}_tfrmattr*' # The unique string in the filepath name based on custom attributes created by RaFTS users



#%%                      CUSTOM ATTRIBUTE AGGREGATION
# Function to convert a string representing a function name into a function object
def _get_function_from_string(func_str: str) -> Callable:
    module_name, func_name = func_str.rsplit('.', 1)  # Split into module and function
    module = globals().get(module_name)               # Get module object from globals()
    if module:
        return getattr(module, func_name)             # Get function object from module


def _subset_ddf_parquet_by_comid(dir_db_attrs: str | os.PathLike,
                                  comid:str,
                                  fp_struct:str = f'*_{comid}_attr*'
                                  ) -> dd.DataFrame:
    """ Read a lazy dataframe corresponding to a single location (comid)

    :param dir_db_attrs: Directory where parquet files of attribute data
      stored
    :type dir_db_attrs: str | os.PathLike
    :param comid: The NHD common identifier (used in filename)
    :type comid: str
    :param fp_struct: f-string formatted unique substring for filename of 
    parquet file corresponding to single location, defaults to f'*_{comid}_*'
    :type fp_struct: str, optional
    :return: lazy dask dataframe of all attributes corresponding to the 
    single comid
    :rtype: dd.DataFrame
    """
    # Based on the structure of comid
    fp = list(Path(dir_db_attrs).rglob(fp_struct) )
    all_attr_ddf = dd.read_parquet(fp, storage_options = None)
    return all_attr_ddf

def _sub_tform_attr_ddf(all_attr_ddf: dd.DataFrame, 
                        retr_vars: str | Iterable, 
                        func: Callable[[Iterable[float]]]) -> np.float:
    """Transform attributes using aggregation function

    :param all_attr_ddf: Lazy attribute data corresponding to a single location (comid)
    :type all_attr_ddf: dd.DataFrame
    :param retr_vars: The basin attributes to retrieve and aggregate by the
      transformation function
    :type retr_vars: str | Iterable
    :param func: The function used to perform the transformation on the `retr_vars`
    :type func: Callable[[Iterable[float]]]
    :return: Aggregated attribute value
    :rtype: np.float
    """
    sub_attr_ddf= all_attr_ddf[all_attr_ddf['attribute'].isin(retr_vars)]
    attr_val = sub_attr_ddf['value'].map_partitions(func, meta=('value','float64')).compute()
    return attr_val

def _cstm_data_src(tform_type: str,retr_vars: str | Iterable) -> str:
    """Standardize the str representation of the transformation function
    For use in the 'data_source' column in the parquet datasets.

    :param tform_type: The transformation function, provided as a str 
    of a simple function (e.g. 'np.mean', 'max', 'sum') for aggregation
    :type tform_type: str
    :param retr_vars: The basin attributes to retrieve and aggregate by the
      transformation function
    :type retr_vars: str | Iterable
    :return: A str representation of the transformation function, with variables
    sorted by character.
    :rtype: str
    """
    # Sort the retr_vars
    retr_vars_sort = sorted(retr_vars)
    return f"{tform_type}([{','.join(retr_vars_sort)}])"


def _gen_tform_df(all_attr_ddf: dd.DataFrame, new_var_id: str,
                    attr_val:np.float, tform_type: str,
                    retr_vars: str | Iterable) -> pd.DataFrame:
    """Generate standard dataframe for a custom transformation on attributes
      for a single location (basin)

    :param all_attr_ddf: All attributes corresponding to a single comid
    :type all_attr_ddf: dd.DataFrame
    :param new_var_id: Name of the newly desired custom variable
    :type new_var_id: str
    :param attr_val: _description_
    :type attr_val: np.float
    :param tform_type: The transformation function, provided as a str 
    of a simple function (e.g. 'np.mean', 'max', 'sum') for aggregation
    :type tform_type: str
    :param retr_vars: The basin attributes to retrieve and aggregate by the
      transformation function
    :type retr_vars: str | Iterable
    :raises ValueError: When the provided dask dataframe contains more than
     one unique location identifier in the 'featureID' column.
    :return: A long-format dataframe of the new transformation variables 
    for a single location
    :rtype: pd.DataFrame
    .. seealso::
        The `proc.attr.hydfab` R package and the `proc_attr_wrap` function
        that generates the standardized attribute parquet file formats
    """
    if all_attr_ddf['featureID'].nunique().compute() != 1:
        raise ValueError("Only expecting one unique location identifier. Reconsider first row logic.")
    
    base_df=all_attr_ddf.iloc[0].compute() # Just grab the first row of a data.frame corresponding to a  and reset the values that matter
    base_df.loc['attribute'] = new_var_id
    base_df.loc['value'] = attr_val
    base_df.loc['data_source'] = _cstm_data_src(tform_type,retr_vars)
    base_df.loc['dl_timestamp'] = datetime.now(datetime.timezone.utc)
    return base_df

# TODO check fp_struct with _attr and w/o _attr once _tformattr written
# Retrieve the variables for a given location (a dask data.frame)
all_attr_ddf = _subset_ddf_parquet_by_comid(dir_db_attrs=dir_db_attrs,
                                            comid=comid,
                                            fp_struct=f'*_{comid}_attr*')

# TODO consider creating a tfrm_cfg parser
def _check_cstm_attr_exst(all_attr_ddf: dd.DataFrame,tfrm_cfg:list,
                          match_method = ['variable','datasource',None][0:2]):
    


    # Generate a list of all custom variables of interest
    ls_cstm_func = list()
    ls_all_cstm_vars = list()
    for item in tfrm_cfg[idx_tfrm_attrs]['transform_attrs']:
        for key, value in item.items():
            ls_tfrm_keys = list(itertools.chain(*[[*x.keys()] for x in value]))
            idx_tfrm_type = ls_tfrm_keys.index('tform_type')
            tfrm_types = value[idx_tfrm_type]['tform_type']
            idx_vars = ls_tfrm_keys.index('vars')
            retr_vars = value[idx_vars]['vars']
            for tform_type in tfrm_types:
                new_var_id = key.format(tform_type=tform_type)
                ls_all_cstm_vars.append(new_var_id)
                ls_cstm_func.append(_cstm_data_src(tform_type,retr_vars))

    sub_attr_need = all_attr_ddf.copy()
    if any([x=='variable' for x in match_method]):
        # Find which variables have already been created:
        subattr_ddf = all_attr_ddf[all_attr_ddf['attribute'].isin(ls_all_cstm_vars)]
        subattrs_avail = subattr_ddf['attribute'].unique().collect() # The attributes already present
        sub_attr_need = sub_attr_need[~sub_attr_need['attribute'].isin(ls_all_cstm_vars)]
    if any([x=='datasource' for x in match_method]):
        # Search which custom datasources (aka the function and variables) match
        subfunc_ddf = all_attr_ddf[all_attr_ddf['data_source'].isin(ls_cstm_func)]
        subfuncs_avail = subfunc_ddf['attribute'].unique().collect()
        sub_attr_need = sub_attr_need[~sub_attr_need['data_source'].isin(ls_cstm_func)]
    # The attributes already present

# TODO identify the names of the desired variables, find which ones don't exist, then only perform transformation and writing if the custom attribute doesn't already exist in the data


def proc_tfrm_cfg(tfrm_cfg: list, idx_tfrm_attrs: int,
                   all_attr_ddf: dd.DataFrame) -> pd.DataFrame:

    # Parse each item in attribute transformation yaml config
    ls_df_rows = []
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

                # Convert string to a function
                func = _get_function_from_string(tform_type)

                # Subset data to variables and compute new attribute
                attr_val = _sub_tform_attr_ddf(all_attr_ddf=all_attr_ddf, 
                            retr_vars=retr_vars, func = func)
                
                # Populate new values in the new dataframe
                new_df = _gen_tform_df(all_attr_ddf=all_attr_ddf, 
                                    new_var_id=new_var_id,
                                    attr_val=attr_val,
                                    tform_type = tform_type,
                                    retr_vars = retr_vars)
                
                ls_df_rows.append(new_df)

    df_new_vars = pd.DataFrame(ls_df_rows)
    return df_new_vars

    