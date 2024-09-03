'''
Unit testing for AlgoTrainEval class in the fs_algo package

example
> coverage run -m unittest test_algo_train_eval.py  
> coverage report
> coverage html 

 # Useful for running in ipynb:
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

'''
import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tempfile
from pathlib import Path
from fs_algo.fs_algo_train_eval import AlgoTrainEval, AttrConfigAndVars
from fs_algo import fs_algo_train_eval
import warnings
import xarray as xr
# %% UNIT TESTING FOR NHDplus/dataset munging
# class TestFsReadAttrComid(unittest.TestCase):
   


# class TestFsRetrNhdpComids(unittest.TestCase):
#     @patch('fs_algo.fs_alg_train_eval.nldi.navigate_byid')

# class TestCheckAttributesExist(unittest.TestCase):
#     print("Testing _check_attributes_exist")
#     # Mock DataFrame
#     mock_pdf = pd.DataFrame({
#         'data_source': 'hydroatlas__v1',
#         'dl_timestamp': '2024-07-26 08:59:36',
#         'attribute': ['pet_mm_s01', 'cly_pc_sav'],
#         'value': [58, 21],
#         'featureID': '1520007',
#         'featureSource': 'COMID'
#     })
#     fs_algo_train_eval._check_attributes_exist(mock_pdf, ['pet_mm_s01','cly_pc_sav'])



# %% UNIT TESTING FOR AttrConfigAndVars

class TestAttrConfigAndVars(unittest.TestCase):
    print("Testing AttrConfigAndVars")
    @patch('builtins.open', new_callable=mock_open, read_data='''
            attr_select:
            - attr_vars: [attr1, attr2, attr3]
            file_io:
            - dir_base: "{home_dir}/base_dir"
            - dir_db_attrs: "{dir_base}/db_attrs"
            - dir_std_base: "{dir_base}/std_base"
            formulation_metadata:
            - datasets: ["dataset1", "dataset2"]
                ''')
    @patch('pathlib.Path.home', return_value='/mocked/home')
    def test_read_attr_config(self, mock_home, mock_file):
        print('    Testing _read_attr_config')
        path = '/path/to/config.yaml'
        attr_obj = AttrConfigAndVars(path)
        attr_obj._read_attr_config()

        # Test if the file is opened with the correct path
        mock_file.assert_called_once_with(path, 'r')

        # Test if Path.home() was called
        mock_home.assert_called_once()

        # Test the parsed data from the config
        expected_attrs_cfg_dict = {
            'attrs_sel': ['attr1', 'attr2', 'attr3'],
            'dir_db_attrs': '/mocked/home/base_dir/db_attrs',
            'dir_std_base': '/mocked/home/base_dir/std_base',
            'dir_base': '/mocked/home/base_dir',
            'datasets': ['dataset1', 'dataset2']
        }

        self.assertEqual(attr_obj.attrs_cfg_dict, expected_attrs_cfg_dict)
    
class TestFsReadAttrComid(unittest.TestCase):   
    @patch('fs_algo.fs_algo_train_eval.dd.read_parquet')
    def test_fs_read_attr_comid(self, mock_read_parquet):
        print("    Testing fs_read_attr_comid")
        # Mock DataFrame
        mock_pdf = pd.DataFrame({
            'data_source': 'hydroatlas__v1',
            'dl_timestamp': '2024-07-26 08:59:36',
            'attribute': ['pet_mm_s01', 'cly_pc_sav'],
            'value': [58, 21],
            'featureID': '1520007',
            'featureSource': 'COMID'
        })
        mock_ddf = dd.from_pandas(mock_pdf, npartitions=1)
        mock_read_parquet.return_value = mock_ddf
        # A normal result scenario, mocked:
        dir_db_attrs = 'mock_dir'
        comids_resp = ['1520007']
        attrs_sel = 'all'
        
        result = fs_algo_train_eval.fs_read_attr_comid(dir_db_attrs=dir_db_attrs, comids_resp=comids_resp, attrs_sel=attrs_sel)

        # Assertions
        self.assertTrue(mock_read_parquet.called)
        self.assertEqual(result.shape[0], 2)
        self.assertIn('1520007', result['featureID'].values)
        self.assertIn('pet_mm_s01', result['attribute'].values)
        self.assertIn('COMID',result['featureSource'].values )
        self.assertIn('value',result.columns )
        self.assertIn('data_source',result.columns )

        # When only one attribute requested
        single_result = fs_algo_train_eval.fs_read_attr_comid(dir_db_attrs=dir_db_attrs,
                                                              comids_resp= comids_resp,attrs_sel= ['pet_mm_s01'])
        self.assertIn('pet_mm_s01',single_result['attribute'].values)
        self.assertNotIn('cly_pc_sav',single_result['attribute'].values)

        # When COMID requested that doesn't exist
        with self.assertWarns(UserWarning):
                fs_algo_train_eval.fs_read_attr_comid(dir_db_attrs=dir_db_attrs,
                                                              comids_resp= ['010101010'],
                                                              attrs_sel= ['pet_mm_s01'])

        # When attribute requested that doesn't exist
        with self.assertWarns(UserWarning):
            fs_algo_train_eval.fs_read_attr_comid(dir_db_attrs=dir_db_attrs,
                                            comids_resp= comids_resp,
                                            attrs_sel= ['nonexistent'])
            

class TestCheckAttributesExist(unittest.TestCase):
    print('Testing _check_attributes_exist')
    def test_check_attributes_exist(self):
        mock_pdf = pd.DataFrame({
            'data_source': 'hydroatlas__v1',
            'dl_timestamp': '2024-07-26 08:59:36',
            'attribute': ['pet_mm_s01', 'cly_pc_sav','pet_mm_s01', 'cly_pc_sav'],
            'value': [58, 21, 65,32],
            'featureID': ['1520007','1520007','1623207','1623207'],
            'featureSource': 'COMID'
            })
        
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter("always")
            fs_algo_train_eval._check_attributes_exist(mock_pdf,pd.Series(['pet_mm_s01','cly_pc_sav']))
            self.assertEqual(len(w),0)

        mock_pdf_bad = mock_pdf.copy()
        mock_pdf_bad.drop(index=0, inplace = True)
        with self.assertWarns(UserWarning):
            fs_algo_train_eval._check_attributes_exist(mock_pdf_bad,pd.Series(['pet_mm_s01','cly_pc_sav']))
        
        # # with warnings.catch_warnings(record = True) as w:
        # #     warnings.simplefilter("always")
        # #     fs_algo_train_eval._check_attributes_exist(mock_pdf_bad,pd.Series(['pet_mm_s01','cly_pc_sav']))
        # #     self.assertEqual(len(w),1)
        
        # mock_pdf_bad1 = mock_pdf.copy()
        # mock_pdf_bad1['attribute'] = 'cly_pc_sav'

        # with self.assertWarns(UserWarning):
        #     fs_algo_train_eval._check_attributes_exist(mock_pdf_bad1,pd.Series(['pet_mm_s01','cly_pc_sav']))
        
        
        # with warnings.catch_warnings(record = True) as w:
        #     warnings.simplefilter("always")
        #     fs_algo_train_eval._check_attributes_exist(mock_pdf_bad1,pd.Series(['pet_mm_s01','cly_pc_sav']))
        #     self.assertEqual(len(w),1)

    # @patch('fs_algo.fs_algo_train_eval.yaml.safe_load')
    # def test_no_attrs_warning(self, mock_home, mock_file):
    #     mock_dict_attr_config = {'attr_select': {'attr_vars': ['attr1','attr2']},
    #                              'file_io': {'dir_base': '/path/dir/base',
    #                                          'dir_db_attrs':'/path/dir/db/attrs',
    #                                          'dir_std_base':'/path/dir/std/base'},
    #                             'formulation_metadata':{'datasets':'test_ds'}}

    # # @patch('pathlib.Path.home', return_value='/mocked/home')
    # def test_no_attrs_warning(self, mock_home, mock_file):
    #     path = '/path/to/config.yaml'
    #     attr_obj = AttrConfigAndVars(path)

        # TODO investigate this here. Began failing after 
        # # Test if the Warning is raised correctly
        # with self.assertWarns(Warning):
        #     attr_obj._read_attr_config()

        # Verify the default behavior when no attributes are selected
        # self.assertEqual(attr_obj.attrs_cfg_dict['attrs_sel'], 'all')


class TestFsRetrNhdpComids(unittest.TestCase):

    def test_fs_retr_nhdp_comids(self):

        # Define test inputs
        featureSource = 'nwissite'
        featureID = 'USGS-{gage_id}'
        gage_ids = ["01031500", "08070000"]

        result = fs_algo_train_eval.fs_retr_nhdp_comids(featureSource, featureID, gage_ids)

        # Assertions
        self.assertEqual(result, ['1722317', '1520007'])

class TestFindFeatSrceId(unittest.TestCase):

    def test_find_feat_srce_id(self):
        attr_config = {'col_schema': [{'featureID': 'USGS-{gage_id}'},
                        {'featureSource': 'nwissite'}],
                        'loc_id_read': [{'gage_id': 'gage_id'},
                        {'loc_id_filepath': '{dir_std_base}/juliemai-xSSA/eval/metrics/juliemai-xSSA_Raven_blended.csv'},
                        {'featureID_loc': 'USGS-{gage_id}'},
                        {'featureSource_loc': 'nwissite'}],
                        }
        rslt = fs_algo_train_eval._find_feat_srce_id(attr_config = attr_config)
        self.assertEqual(rslt,['nwissite','USGS-{gage_id}'])

    # Raise error when featureSource not provided:
    def test_missing_feat_srce(self):
        attr_config_miss = {'col_schema': [{'featureID': 'USGS-{gage_id}'},
                                            {'fe0a0ur0eS0ou0r0ce': 'nwissite'}],
                            'loc_id_read': {'gage_id': 'gage_id'}}
        with self.assertRaises(ValueError):
            fs_algo_train_eval._find_feat_srce_id(attr_config = attr_config_miss, dat_resp=None)

    def test_netcdf_attributes(self):
         # Create a mock xarray.Dataset object w/ attributes
        mock_xr = MagicMock(spec=xr.Dataset)
        mock_xr.attrs = {'featureSource': 'nwissite',
                         'featureID': 'USGS-{gage_id}'}

        rslt = fs_algo_train_eval._find_feat_srce_id(mock_xr)
        self.assertEqual(rslt,['nwissite','USGS-{gage_id}'])


    # Raise error when featureID not provided:
    def test_missing_feat_id(self):
         # Create a mock xarray.Dataset object
        mock_xr = MagicMock(spec=xr.Dataset)
        mock_xr.attrs = {'featureSource': 'nwissite',
                         'f0e1a0tu1reID': 'USGS-{gage_id}'}

        with self.assertRaises(ValueError):
            fs_algo_train_eval._find_feat_srce_id(mock_xr)

    # Test when dataset does not have any attributes but does have config:
    def test_missing_attrs(self):
        mock_xr = MagicMock(spec=xr.Dataset)
        mock_xr.attrs = {'notit': 'blah',
                         'alsonotit': 'bleh'}
        attr_config = {'col_schema': [{'featureID': 'USGS-{gage_id}'},
                {'featureSource': 'nwissite'}],
                'loc_id_read': [{'gage_id': 'gage_id'},
                {'loc_id_filepath': '{dir_std_base}/juliemai-xSSA/eval/metrics/juliemai-xSSA_Raven_blended.csv'},
                {'featureID_loc': 'USGS-{gage_id}'},
                {'featureSource_loc': 'nwissite'}],
                }
        rslt = fs_algo_train_eval._find_feat_srce_id(dat_resp = mock_xr, attr_config = attr_config)
        self.assertEqual(rslt,['nwissite','USGS-{gage_id}'])

class TestFsSaveAlgoDirStruct(unittest.TestCase):
    def test_fs_save_algo_dir_struct(self):
        dir_base = tempfile.gettempdir()
        rslt = fs_algo_train_eval.fs_save_algo_dir_struct(dir_base)
        self.assertIn('dir_out', rslt.keys())
        self.assertIn('dir_out_alg_base', rslt.keys())
        self.assertTrue(Path(rslt['dir_out_alg_base']).exists)

        with self.assertRaises(ValueError):
            fs_algo_train_eval.fs_save_algo_dir_struct(dir_base + '/not_a_dir/')

class TestOpenResponseDataFsds(unittest.TestCase):
    dir_std_base = tempfile.gettempdir()

    def test_open_response_data_fsds(self):

        with self.assertRaisesRegex(ValueError, 'Could not identify an approach to read in dataset'):
            fs_algo_train_eval._open_response_data_fsds(self.dir_std_base,ds='not_a_ds')


# %% UNIT TEST FOR AlgoTrainEval class
class TestAlgoTrainEval(unittest.TestCase):
    print("Testing AlgoTrainEval")
    def setUp(self):
        # Create a simple DataFrame for testing
        data = {
            'attr1': [1, 2, 3, 4, 5],
            'attr2': [5, 4, 3, 2, 1],
            'metric1': [0.1, 0.9, 0.3, 0.1, 0.8]
        }
        self.df = pd.DataFrame(data)

        # Variables and configurations for algorithms
        self.attrs = ['attr1', 'attr2']
        self.algo_config = {
            'rf': {'n_estimators': 10},
            'mlp': {'hidden_layer_sizes': (10,), 'max_iter': 200}
        }
        self.dataset_id = 'test_dataset'
        self.metric = 'metric1'

        # Output directory
        self.dir_out_alg_ds = tempfile.gettempdir()

        # Instantiate AlgoTrainEval class
        self.train_eval = AlgoTrainEval(df=self.df, attrs=self.attrs, algo_config=self.algo_config,
                                 dir_out_alg_ds=self.dir_out_alg_ds, dataset_id=self.dataset_id,
                                 metr=self.metric, test_size=0.4, rs=42)

    def test_split_data(self):
        # Test data splitting
        self.train_eval.split_data()
        self.assertEqual(len(self.train_eval.X_train), 3)
        self.assertEqual(len(self.train_eval.X_test), 2)
        self.assertEqual(len(self.train_eval.y_train), 3)
        self.assertEqual(len(self.train_eval.y_test), 2)

    def test_train_algos(self):
        # Test algorithm training
        self.train_eval.split_data()
        self.train_eval.train_algos()

        self.assertIn('rf', self.train_eval.algs_dict)
        self.assertIsInstance(self.train_eval.algs_dict['rf']['algo'], RandomForestRegressor)

        self.assertIn('mlp', self.train_eval.algs_dict)
        self.assertIsInstance(self.train_eval.algs_dict['mlp']['algo'], MLPRegressor)

        #self.assertEqual(len(self.algo_config), len(self.train_eval))

    def test_predict_algos(self):
        # Test algorithm predictions
        self.train_eval.split_data()
        self.train_eval.train_algos()

        preds = self.train_eval.predict_algos()

        self.assertIn('rf', preds)
        self.assertIn('mlp', preds)
        self.assertEqual(len(preds['rf']['y_pred']), 2)  # Number of test samples
        self.assertEqual(len(preds['mlp']['y_pred']), 2)

    def test_evaluate_algos(self):
        # Test evaluation of algorithms
        self.train_eval.split_data()
        self.train_eval.train_algos()
        self.train_eval.predict_algos()

        eval_dict = self.train_eval.evaluate_algos()

        self.assertIn('rf', eval_dict)
        self.assertIn('mlp', eval_dict)
        self.assertIn('mse', eval_dict['rf'])
        self.assertIn('r2', eval_dict['mlp'])

    @patch('joblib.dump')
    def test_save_algos(self, mock_dump):
        # Test saving algorithms to disk
        self.train_eval.split_data()
        self.train_eval.train_algos()

        # Mock joblib.dump to avoid file operations
        self.train_eval.save_algos()
        self.assertTrue(mock_dump.called)

        for algo in self.train_eval.algs_dict.keys():
            self.assertIn('loc_pipe', self.train_eval.algs_dict[algo])

    def test_org_metadata_alg(self):
        # Test organizing metadata
        self.train_eval.split_data()
        self.train_eval.train_algos()
        self.train_eval.predict_algos()
        self.train_eval.evaluate_algos()

        # Mock saving algorithms and call organization
        with patch('joblib.dump'):
            self.train_eval.save_algos()

        self.train_eval.org_metadata_alg()

        # Check eval_df is correctly populated
        self.assertFalse(self.train_eval.eval_df.empty)
        self.assertIn('dataset', self.train_eval.eval_df.columns)
        self.assertIn('loc_pipe', self.train_eval.eval_df.columns)
        self.assertIn('algo', self.train_eval.eval_df.columns)
        self.assertEqual(self.train_eval.eval_df['dataset'].iloc[0], self.dataset_id)

    def test_train_eval(self):
        # Test the overall wrapper method
        with patch('joblib.dump'):
            self.train_eval.train_eval()

        # Check all steps completed successfully
        self.assertFalse(self.train_eval.eval_df.empty)
        self.assertIn('rf', self.train_eval.algs_dict)
        self.assertIn('mlp', self.train_eval.algs_dict)
        self.assertIn('rf', self.train_eval.preds_dict)
        self.assertIn('mlp', self.train_eval.preds_dict)
        self.assertIn('rf', self.train_eval.eval_dict)
        self.assertIn('mlp', self.train_eval.eval_dict)


if __name__ == '__main__':
    unittest.main()