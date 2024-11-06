'''
Functions for loading custom attributes data. Uses CAMELS 
    and GAGES-II attributes for testing.
author:: Benjamin Choat <benjamin.choat@noaa.gov>
description:: given usgs catchment ID's, attributes are read in for
    those ID's.
notes:: developed using python v3.11.8; created 2024/08/02
'''

# TODO:
# - metadata (user provied)
# - unit tests
# error or warning if id's are provided that do not appear in data
# modfiy to write out 1 parquet for each id

# import libraries needed across all functions
import warnings
import os
# from glob import glob (Keith is gone)
# from pathlib import Path
# import shutil
import pandas as pd
import yaml
# from fsds_proc import data
from openpyxl import load_workbook
# from openpyxl.utils.exceptions import InvalidFileException


######################################################################

def check_config_valid(config_main):
    '''
    Check that configuration file is valid

    Inputs
    ----------------------
    config_main ()
    '''
    # check 1st layer of keys seem correct

    # define list of expected keys
    keys_expected = [
        'dir_output',
        'files_in',
        'file_id_subsets',
        'attribute_metadata',
        'references'
        ]

    # read 1st layer keys
    keys_1st = list(config_main.keys())

    # if keys_expected and keys_1st don't match, check which keys are different
    if keys_expected != keys_1st:
        in_expected = [x for x in keys_expected if x not in keys_1st]
        in_temp = [x for x in keys_1st if x not in keys_expected]

        if len(in_expected) == 1 and in_expected[0] != 'file_id_subsets':
            message_out1 = (
                "\nThe following key is missing at the first level in the "
                "configuration file:\n"
                f"{in_expected}."
            )

        elif len(in_expected) > 1:
            message_out1 = (
                "\nThe following key is missing at the first level in the "
                "configuration file:\n"
                f"{in_expected}."
            )
        else:
            message_out1 = ""

        if len(in_temp) > 0:
            message_out2 = (
                "\nThe following keys were provided at the first level in "
                "the configuration file, but should not be:\n"
                f"    {in_temp}\n"
            )
        else:
            message_out2 = ""

        if (len(in_expected) == 1 and in_expected[0] != 'file_id_subsets' or
                len(in_expected) > 1 or len(in_temp) > 0):
            raise ValueError(f'{message_out1}\n{message_out2}')

    if config_main['dir_output'] is None:
        message_out = (
            "No value was provided for dir_output. It should specify "
            "where to place output files."
        )

        raise ValueError(message_out)

    # TODO: Prolly remove since outputing statndard parquet files
    # out_extension = config_main['file_output'].split('.')[
    #     len(config_main['file_output'].split('.'))-1
    #     ]

    # if out_extension not in ['csv', 'parquet']:
    #     message_out = (
    #         "Allowed file extensions for the output file are .csv or\n"
    #         f"    .parquet. You provided '.{out_extension}'."
    #     )
    #     raise ValueError(message_out)s

    # check if csv with subset of IDs provided. If so, check extension is csv
    if 'file_id_subsets' in keys_1st:
        id_temp = config_main['file_id_subsets']
        id_extension = id_temp.split('.')[len(id_temp.split('.'))-1]
        if id_extension != 'csv':
            raise ValueError(
                f'You provided {id_temp}\n    as a file holding gauge IDs to'
                f' be used, but that file must have a .csv extenstion.'
            )

    # check input files seem correct

    # get listed files

    # define list of required keys for .txt/.csv and .xlsx/.xlsm
    txtcsv_accepted = [
        'separator', 'id_column', 'attr_dataset_id', 'attr_columns'
        ]
    xl_accepted = ['tabs', 'id_column', 'attr_dataset_id', 'attr_colums']

    listed_files = list(config_main['files_in'].keys())

    # messages will be updated as errors are found
    message_out1 = ''
    message_out2 = ''

    for file in listed_files:
        extension = file.split('.')[len(file.split('.'))-1]
        keys_file = list(config_main['files_in'][file].keys())
        print(f'keys_file: {keys_file}')
        if extension in ['txt', 'csv']:
            in_expected = [x for x in txtcsv_accepted if x not in keys_file]
            in_temp = [x for x in keys_file if x not in txtcsv_accepted]

            if len(in_expected) == 1 and in_expected[0] == 'separator':
                warnings.warn(
                    f"\nseparator was not provided for {file}\n"
                    f"  so the code will try to detect it automatically."
                )
            if ((len(in_expected) == 1 and in_expected[0] != 'separator') or
                    len(in_expected) > 1):
                message_temp = (
                    'You did not provide the following expected keys for \n'
                    f'  {file}:\n    {in_expected}'
                )
                message_out1 = f'{message_out1}{message_temp}'
            if len(in_temp) > 0:
                message_temp = (
                    "You provided the following keys in the configuration\n"
                    f"  file, for {file}\n"
                    f"  but they are not accepted:\n{in_temp}\n"
                )
                message_out2 = message_temp

        elif extension in ['xlsx', 'xlsm']:
            in_expected = [x for x in xl_accepted if x not in keys_file]
            in_xl = [x for x in keys_file if x not in xl_accepted]

            if len(in_expected) == 1 and in_expected[0] == 'tabs':
                warnings.warn(
                    f"\ntabs were not provided for {file} so data in\n"
                    "the first tab will be used."
                )
            if ((len(in_expected) == 1 and in_expected[0] != 'tabs') or
                    len(in_expected) > 1):
                message_temp = (
                    'You did not provide the following expected keys for \n'
                    f'  {file}:\n    {in_expected}\n'
                )
                message_out2 = message_temp
            else:
                message_out2 = ''
        else:
            raise ValueError(
                    f"Incorrect extension ({extension}).\n"
                    "Allowed file extensions for input files include "
                    "[.txt, .csv, .xlsx, .xlsm]"
                )

        if len(message_out1) > 0 or len(message_out2) > 0:
            raise ValueError(f'{message_out1}\n{message_out2}')

        # handle subkeys of tabs if present
        if 'tabs' in keys_file:
            temp_dict = config_main['files_in'][file]['tabs']
            tabs_expected = ['id_column', 'attr_columns']
            tabs_allowed = tabs_expected.copy()
            tabs_allowed.extend(['attr_dataset_id'])
            for tab in temp_dict.keys():
                keys_tab = list(temp_dict[tab].keys())
                in_expected = [
                    x for x in tabs_expected if x not in keys_tab
                    ]
                in_tab = [x for x in keys_tab if x not in tabs_allowed]

                # handle case where attr_dataset_id provided for file and tab
                print(f'keys_tab: {keys_tab}')
                print(f'in_tab: {in_tab}')
                if ('attr_dataset_id' in keys_file and
                        'attr_dataset_id' in keys_tab):
                    warnings.warn(
                        'attr_dataset_id was provided for the file and\n'
                        f'   tab. The id provided for the tab will be used.\n'
                        f"  file: {list(config_main['files_in'].keys())[0]}\n"
                        f'  tab: {tab}'
                    )

                # handle case where attr_dataset_id not provided at all
                if ('attr_dataset_id' not in keys_file and
                        'attr_dataset_id' not in keys_tab):
                    message_out = (
                        'You must provide attr_dataset_id for each file.\n'
                        '   You may proivde a different attr_dataset_id for\n'
                        '   each tab in an excel file.\n'
                        )

                    raise ValueError(message_out)

                if len(in_expected) > 0:
                    message_out1 = (
                        f"The following keys are missing for {file}:\n"
                        f"   {in_expected}"
                        )
                else:
                    message_out1 = ''

                if len(in_tab) > 0:
                    message_out2 = (
                        f"The following keys were provided for {file}\n but "
                        f"    were not expected:\n{in_tab}"
                    )
                else:
                    message_out2 = ''

                if len(in_expected) > 0 or len(in_tab) > 0:
                    raise ValueError(f'{message_out1}\n{message_out2}')

    return None


######################################################################


def check_columns_present(dict_config: dict,
                          df_data: pd.DataFrame,
                          gage_id_col: str,
                          file: str | os.PathLike) -> str:
    '''
    Function checks that columns specified in dict_config are present
    in df_data. First checks if id column is present, then checks
    if attribute columns are present.
    If specified columns are not present then an error is thrown.
    
    Inputs
    -------------------
    dict_config (dict): dictionary generated when custom_attribute_config.yaml
        is read in. 
    df_data (pandas DataFrame): a pandas dataframe expected to contain columns
        with a gage_id column name and attribute column names specified in 
        dict_config
    gage_id_col (str): Name of column that holds values to be used as IDs
    file (str): name of file from which df_data was read in

    Outputs
    -------------------
    message_id (str): Not intneded to be used as output, but instead printed
        during processing
    message_attr (str): Not intneded to be used as output, but instead printed
        during processing

    '''
    # check specified gage_id column is present in current file
    if gage_id_col not in df_data.columns:
        raise ValueError(
            f"In the configuration file '{gage_id_col}'\n"
            "   was specified to be the name of the\n"
            "   id column, but there does not"
            f"  seem to be a column\nwith that name in {file}."
        )

    else:
        message_id = f'\n"{gage_id_col}" used as gage_id column in {file}.\n'

    # id columns listed in config yaml if they are not in current file
    missing_config_cols = [
        x for x in dict_config['attr_columns'] if x not in df_data.columns
    ]

    # if there are missing columns, return error
    if len(missing_config_cols) > 0:
        raise ValueError(
            "In the configuration yaml file the following\n"
            f"  columns were listed, but they do not appear in\n"
            f"  {file}:\n{missing_config_cols}"
        )
    else:
        message_attr = f"\nAll specified columns for {file} are present\n"

    print(message_id)
    print(message_attr)
    return message_id, message_attr


######################################################################


def process_txt_csv(dict_config: dict,
                    file: str | os.PathLike,
                    ids_subset: list = None) -> pd.DataFrame:
    '''
    Read in file and process according to inputs in dict_config

    Inputs
    ----------------
    dict_config (dict): dictionary generated when custom_attribute_config.yaml
        is read in and subset to specific file in files_in. 
    file (str): name of file from which data will be read in
    ids_subset (list): list of gauge ids used to subset data

    Outputs
    ----------------
    df (pd.DataFrame): Dataframe containing data from file and processed 
        according to dict_config
    '''

    # get gage_id column name so it can be read in as string in case
    # there are leading 0's
    gage_id_temp = dict_config['id_column']

    # get dataset id
    attr_data_id = dict_config['attr_dataset_id']

    # extract separator if provided
    if 'separator' in dict_config and dict_config['separator']:
        if len(dict_config['separator']) != 1:
            raise ValueError(
                "Only 1 separator is allowed."
            )
        separator = dict_config['separator']

        # read in file
        try:
            df = pd.read_csv(
                file,
                sep=separator,
                dtype={gage_id_temp: str},
                usecols=[gage_id_temp] + dict_config['attr_columns']
                )
        except Exception as exc:
            raise exc
    else:
        warnings.warn(
            f"\nseparator was not provided for \n{file} in config "
            " yaml file. \n    so attempting to identify automatically."
        )

        # sep=None uses python csv sniffer to id separator
        try:
            df = pd.read_csv(
                file,
                sep=None,
                engine='python',
                dtype={gage_id_temp: str},
                usecols=[gage_id_temp] + dict_config['attr_columns']
                )
        except Exception as exc:
            raise exc

    if ids_subset and len(ids_subset) > 0:
        df = df.query(f"{gage_id_temp} in {ids_subset}")

    dict_temp = {
        x: attr_data_id for x in df.columns if x != gage_id_temp
    }

    dict_temp['source_file'] = os.path.basename(file)

    # check that listed columns match current file columns
    check_columns_present(dict_config, df, gage_id_temp, file)

    return df, dict_temp

######################################################################


def process_excel(dict_config: dict,
                  file: str | os.PathLike,
                  tabs_included: bool,
                  ids_subset: list = None) -> tuple[pd.DataFrame, list]:
    '''
    Read in file and process according to inputs in dict_config

    Inputs
    ----------------
    dict_config (dict): dictionary generated when custom_attribute_config.yaml
        is read in. 
    file (str): name of file from which data will be read in
    tabs_included (boolean): True if tabs provided in config yaml, otherwise False
    ids_subset (list): list of gauge ids used to subset data

    Outputs
    ----------------
    df (pd.DataFrame): Dataframe containing data from file and processed 
        according to dict_config
    gage_id_cols (list): A list of gage_ids associated with each file or tab
    '''

    # load excel workbook
    try:
        excel_wb = load_workbook(
                        filename=file,
                        data_only=True,
                        read_only=True
                        )
    except Exception as exc:
        raise exc

    # get dataset id
    try:
        attr_data_id = dict_config['attr_dataset_id']
    except KeyError:
        warnings.warn(
            f'attr_data_id not provided for {file},\n'
            '   so using value provided for tab.\n'
            )

    if tabs_included:

        # get list of tabs associated with current file
        tabs_temp = list(dict_config['tabs'].keys())

        # create dict to hold gage_id_column for each tab-used for mergin in end
        dict_gg_temp = {x: [] for x in tabs_temp}

        # create empty list to store data from tabs in
        dict_dfs_tabs = {tab: [] for tab in tabs_temp}

        # create temperary dict ot hold data and source info from each tab
        dict_temp = {}

        for j, tab in enumerate(tabs_temp):

            if 'attr_dataset_id' in dict_config['tabs'][tab].keys():
                print(f'\nattr_dataset_id found for {tab}.\n')
                attr_data_id = dict_config['tabs'][tab]['attr_dataset_id']

            # get gage_id column name so it can be read in as string in case
            # there are leading 0's
            gage_id_temp = dict_config['tabs'][tab]['id_column']

            # create or append gage_id col to list to be used later when
            # merging data
            if j == 0:
                gage_id_cols = [gage_id_temp]
            else:
                gage_id_cols.append(gage_id_temp)

            # extract worksheet and convert to df generator
            excel_ws = excel_wb[tab].values
            # store first row as colnames
            colnames = next(excel_ws)

            # create dataframe
            df_tab = pd.DataFrame(excel_ws, columns=colnames)

            check_columns_present(
                dict_config['tabs'][tab],
                df_tab,
                gage_id_temp,
                file
                )

            # subset to desired columns
            cols_keep = (
                [gage_id_temp] + dict_config['tabs'][tab]['attr_columns']
                )

            # add current dataframe to tabs output dict
            dict_dfs_tabs[tab] = df_tab[cols_keep]

            # add gage_id to dict holding gage_id for each tab
            dict_gg_temp[tab] = gage_id_temp

            # merge dataframes using gage_ids
            df = dict_dfs_tabs[tabs_temp[0]]

            # susbet to provided ids, if provided
            if ids_subset and len(ids_subset) > 0:
                df = df.query(f"{gage_id_temp} in {ids_subset}")

            print("IS IT HERE")
            dict_temp_inner = {
                x: attr_data_id for x in df.columns if x != gage_id_temp
            }

            dict_temp_inner['source_file'] = os.path.basename(file)

            dict_temp = {**dict_temp, **dict_temp_inner}

            print("OR IS IT HERE")

        for tab in tabs_temp[1:]:
            df = pd.merge(
                df, dict_dfs_tabs[tab],
                left_on=dict_gg_temp[tabs_temp[0]],
                right_on=dict_gg_temp[tab],
                how='inner'
            )

    # if xlsx or xlsm and tab not included, then assume first tab is
    # desired tab
    else:
        warnings.warn(
            f"\nFor {file}\n    you did not include tabs, so the first "
            "tab in the file\n   will be taken as the desired data. If "
            "this is incorrect, please\n"
            "   specify the tabs you wish to use."
        )

        # get gage_id column name so it can be read in as string in case
        # there are leading 0's
        gage_id_temp = dict_config['id_column']

        gage_id_cols = gage_id_temp

        # extract worksheet and convert to df generator
        excel_ws = excel_wb.worksheets[0].values
        # store first row as colnames
        colnames = next(excel_ws)

        # create dataframe
        df = pd.DataFrame(excel_ws, columns=colnames)

        # susbet to provided ids, if provided
        if ids_subset and len(ids_subset) > 0:
            df = df.query(f"{gage_id_temp} in {ids_subset}")

        check_columns_present(dict_config, df, gage_id_temp, file)

        # subset to desired columns
        cols_keep = [gage_id_temp] + dict_config['attr_columns']

        # subet df to cols_keep
        df = df[cols_keep]

        dict_temp = {
                x: attr_data_id for x in df.columns if x != gage_id_temp
            }

        dict_temp['source_file'] = os.path.basename(file)

    return df, gage_id_cols, dict_temp


######################################################################


def process_custom_attributes(config_file: str | os.PathLike) -> pd.DataFrame:
    '''
    Given information provided in custom_attr_config.yaml, loads
    in custom attributes, standardizes them, and saves them to
    destination as file type specified in custom_attr_config.yaml.

    Inputs
    -------------
    path (str or path): path to folder holding CAMELS data.
    ids (list): list of gauge-ids associated with CAMELS data and
        to be extracted when this function is executed.
        Be careful that leading 0's are present in ids.

    Returns
    --------------
    df (pandas DataFrame):
    '''

    # load YAML config file
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # check configuration file seems correct
    check_config_valid(config)

    # process configuration file

    # check if file with ids for subsetting data are provided
    subset_ids = None
    if ('file_id_subsets' in config.keys() and
            len(config['file_id_subsets']) > 0):
        df_ids = pd.read_csv(config['file_id_subsets'], dtype=str)
        subset_ids = list(df_ids.iloc[:, 0])

    # get list of files
    files_in = list(config['files_in'].keys())

    # define empty dict to append data to as read in
    dict_dfs_in = {f: [] for f in files_in}

    # initiate an empty dictionary to hold information about the file
    # and data id
    dict_data_source = {}

    # process files listed in files_in
    for i, file in enumerate(files_in):

        print(f'Processing file {i+1} of {len(files_in)}\nfile: {file}')

        # store config for file in dict_temp
        dict_temp = config['files_in'][file]

        # get extension to indicate file type
        temp = file.split('.')
        extension = temp[len(temp)-1]

        # read in txt, csv,
        if extension in ['txt', 'csv']:

            # create or append gage_id col to list to be used later when
            # merging data
            if i == 0:
                gage_id_cols = [dict_temp['id_column']]
            else:
                gage_id_cols.append(dict_temp['id_column'])

            df_temp, dict_vars = process_txt_csv(dict_temp, file, subset_ids)

            # update dict holding info about vars and their source
            dict_data_source = {**dict_data_source, **dict_vars}

        else:  # if extension in ['xlsx', 'xlsm']:

            # check if current file included tabs; if not then assume first
            # tab is desired
            tabs_included = True if 'tabs' in config['files_in'][file] \
                else False
            # process the file
            df_temp, gage_id_temp, dict_vars = process_excel(
                                        dict_temp,
                                        file,
                                        tabs_included,
                                        subset_ids
                                    )

            # update dict holding info about vars and their source
            dict_data_source = {**dict_data_source, **dict_vars}

            # create or append gage_id col to list to be used later when
            # merging data
            if i == 0:
                gage_id_cols = gage_id_temp
            else:
                gage_id_cols.extend(gage_id_temp)

        # add current data to dictionary of output dataframes
        dict_dfs_in[file] = df_temp

    # merge all data into a single dataframe
    df_out = dict_dfs_in[files_in[0]]
    for i, file in enumerate(files_in[1:]):
        df_out = pd.merge(
            df_out, dict_dfs_in[file],
            left_on=gage_id_cols[0],
            right_on=gage_id_cols[i+1],
            how='inner'
        )

    # keep only one id column
    if isinstance(gage_id_cols, list) and len(set(gage_id_cols)) > 1:
        df_out = df_out.drop(list(set(gage_id_cols))[1:], axis=1)

    # TODO: modify to write out standardized parquets to dir_out
    # out_extension = config['file_output'].split('.')[
    #     len(config['file_output'].split('.'))-1
    #     ]
    # if out_extension == 'csv':
    #     df_out.to_csv(config['file_output'], index=False)
    # else:
    #     df_out.to_parquet(config['file_output'], index=False)

    # print(f"\nOutput file saved to {config['file_output']}\n")

    # convert dataframe to long format, then loop through and write
    # data for each id to parquet
    print(f'df_out: {df_out}')
    print(f'gage_id_cols: {gage_id_cols[0]}')
    df_out = pd.melt(
        df_out,
        id_vars=gage_id_cols[0],
        value_vars=[x for x in df_out.columns if x != gage_id_cols[0]],
        var_name='ID',
        value_name='Variable'
    )
    # TODO: use dict_Vars to add variable source info to dataframe
    # add data information to dataframe
    # df_out['']

    # write a parquet file for each gauge_id

    return df_out
    # return config


######################################################################


if __name__ == '__main__':
    test1 = process_custom_attributes(
        'tests/custom_attribute_data/custom_attribute_config.yaml'
        )
    # test2 = process_custom_attributes(
    #     'tests/custom_attribute_data/custom_attribute_config2.yaml'
    #     )
    # test3 = process_custom_attributes(
    #     'tests/custom_attribute_data/custom_attribute_config3.yaml'
    #     )

    # df_test = pd.read_parquet(
    # 'tests/custom_attribute_data/TEST_CUSTOM_OUT1.parquet'
    # )
    df_test = pd.read_csv(
        'tests/custom_attribute_data/TEST_CUSTOM_OUT3.csv',
        dtype={'STAID': str}
        )
