'''
Unit tests for the fs_proc package

example::
> cd /path/to/fs_proc/fs_proc/tests/
> python -m unittest test_proc_eval_metrics.py
or if interested in unit testing coverage:
> python -m coverage run -m unittest
> python -m coverage report 
# and may also run the following to generate an html: > python -m coverage html 


notes::
Changelog/contributions
    2024-07-11 Originally created, GL
'''


import unittest
from pathlib import Path
import pandas as pd
import yaml
import tempfile
import xarray as xr
from fs_proc.proc_eval_metrics import read_schm_ls_of_dict, proc_col_schema, _proc_check_input_config, _proc_flatten_ls_of_dict_keys, _proc_check_input_df, _proc_check_std_fs_ids
import numpy as np
from unittest.mock import patch

# Define the unit test directory for fs_proc
parent_dir_test = Path(__file__).parent #TODO should change this 
print(f'Running unit test from {parent_dir_test}')
# Define the unit test saving directory as a temp dir
dir_save = tempfile.gettempdir()

# Load the YAML configuration file from the testing data
schema_dir_test = Path(parent_dir_test/Path("user_data_schema.yaml"))
with open(schema_dir_test, 'r') as file:
    config = yaml.safe_load(file)

# Reads the testing config dataframe
exp_config_df = pd.read_csv(Path(parent_dir_test/Path("test_config_df.csv")), index_col=None)

# Load the user-specific metrics dataset from the testing data
test_df = pd.read_csv(Path(parent_dir_test/Path("user_metric_data.csv")))
raw_test_df = test_df.rename(columns = dict(zip(exp_config_df['metric_mappings'].str.split('|')[0],
    exp_config_df['metric_cols'].str.split('|')[0])))

class TestReadSchmLsOfDict(unittest.TestCase):
    '''
    A normal run. This test needs to be updated (write new df example) anytime the user_data_schema.yaml changes.
    '''
    def test_identical(self):
        global parent_dir_test
        global schema_dir_test
        global exp_config_df
        gen_config_df = read_schm_ls_of_dict(schema_dir_test).fillna(np.nan).infer_objects(copy=False)
        pd.testing.assert_frame_equal(exp_config_df, gen_config_df, check_dtype = False)


class TestProcColSchema(unittest.TestCase):
    '''
    A normal run
    '''
    global raw_test_df
    global dir_save
    global exp_config_df
    ds = proc_col_schema(raw_test_df, exp_config_df, dir_save)

    def test_dataset_type(self):
        self.assertIsInstance(self.ds, xr.Dataset)

    def test_written_dir_exists(self):
        self.assertTrue(Path(dir_save/Path('user_data_std')).resolve().is_dir())

    def test_written_file_exists(self):
        self.assertTrue(Path(dir_save/Path('user_data_std/juliemai-xSSA/eval/metrics/juliemai-xSSA_Raven_blended.csv')).resolve().is_file())

    def test_dataset_vars(self):
        self.assertEqual(list(self.ds.keys()),['basin_name','NSE','RMSE','KGE'])

class TestProcColSchemaHier(unittest.TestCase):
    global raw_test_df
    global dir_save
    global exp_config_df

    # specify netcdf config
    nc_config_df = exp_config_df.copy()
    nc_config_df['save_type'] = 'netcdf'
    dsnc = proc_col_schema(raw_test_df,nc_config_df, dir_save)
    def test_hier_nc_exists(self):
        self.assertTrue(list(Path(dir_save/Path('user_data_std/juliemai-xSSA/')).glob('*.nc'))[0].is_file())

    # specifiy zarr config
    zarr_config_df = exp_config_df.copy()
    zarr_config_df['save_type'] = 'zarr'
    dsz = proc_col_schema(raw_test_df,zarr_config_df, dir_save)
    def test_hier_file_exists(self):
         self.assertTrue(list(Path(dir_save/Path('user_data_std/juliemai-xSSA/')).glob('*/*.zattrs'))[0].is_file())

class TestProcCheckInputDf(unittest.TestCase):
    global test_df
    global raw_test_df
    global exp_config_df

    proc_df = _proc_check_input_df(raw_test_df,exp_config_df)

    metric_cols = exp_config_df['metric_cols'].iloc[0].split('|')
    mapping_cols = exp_config_df['metric_mappings'].iloc[0].split('|')
    def test_expect_warn(self):
        with self.assertWarns(Warning):
            _proc_check_input_df(test_df, exp_config_df)
    def test_expect_change_metric_cols(self):
        self.assertTrue(all([x in self.proc_df.columns for x in self.mapping_cols]))

    def test_expect_warn_two_gage_ids(self):
        self.proc_df['gage_id'] = 'aaa'
        with self.assertWarns(Warning):
            _proc_check_input_df(self.proc_df, exp_config_df)

    def test_expect_warn_missing_col(self):
        bad_test_df = raw_test_df.drop('nse', axis=1)
        with self.assertWarns(Warning):
            _proc_check_input_df(bad_test_df, exp_config_df)

class TestProcCheckStdFsIds(unittest.TestCase):
    def test_notavar_error(self):
        self.assertRaises(ValueError,
                          _proc_check_std_fs_ids,vars=['notavar' ], category = 'metric')
    
    @patch('builtins.print')
    def test_atomic_var(self,mock_print):
        _proc_check_std_fs_ids(vars= 'NSE', category = 'metric')
        mock_print.assert_called_with('The metric mappings from the dataset schema match expected format.')
    
class TestProcCheckInputConfig(unittest.TestCase):
    
    def test_std_keys(self):
        global config
        with self.assertRaises(Exception) as context:
            _proc_check_input_config(config, std_keys=['not the standard keys'])
            self.assertTrue(' provided keys in the input config file' in str(context.exception))
    def test_std_col(self):
        global config
        self.assertRaises(ValueError, 
                          _proc_check_input_config, config, req_col_schema=['not the standard col names'])
    def test_form_meta(self):
        global config
        self.assertRaises(ValueError, 
                          _proc_check_input_config, config, req_form_meta=['not the standard formulation metadata'])
    def test_file_io(self):
        global config
        self.assertRaises(ValueError, 
                          _proc_check_input_config, config, req_file_io=['not the standard dir or save keys'])

class TestProcFlattenLsOfDictKeys(unittest.TestCase):
    global config
    ls_fio = _proc_flatten_ls_of_dict_keys(config, 'file_io')

    def test_return_ls(self):
        self.assertTrue(type(self.ls_fio) == list)

    def test_size_ls(self):
        self.assertTrue(len(self.ls_fio) == 5)


if __name__ == '__main__':
    unittest.main()
