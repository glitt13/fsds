'''
@title: Produce data visualizations for RaFTS model performance outputs
@author: Lauren Bolotin <lauren.bolotin@noaa.gov>
@description: Reads in several config files, 
    visualizes results for the specified RaFTS algorithms and evaluation metrics, 
    and saves plots to .png's.
@usage: python fs_perf_viz.py "/full/path/to/viz_config.yaml"

Changelog/contributions
    2024-11-22 Originally created, LB
'''
import geopandas as gpd
import os
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
import yaml
from pathlib import Path
import argparse
import fs_algo.fs_algo_train_eval as fsate
import xarray as xr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'process the data visualization config file')
    parser.add_argument('path_viz_config', type=str, help='Path to the YAML configuration file specific for data visualization')
    args = parser.parse_args()

    home_dir = Path.home()
    path_viz_config = Path(args.path_viz_config) #Path(f'{home_dir}/FSDS/formulation-selector/scripts/eval_ingest/xssa/xssa_viz_config.yaml') 

    with open(path_viz_config, 'r') as file:
        viz_cfg = yaml.safe_load(file)

    # Get features from the viz config file --------------------------
    algos = viz_cfg.get('algos')
    print('Visualizing data for the following RaFTS algorithms:')
    print(algos)
    print('')
    metrics = viz_cfg.get('metrics')
    print('And for the following evaluation metrics:')
    print(metrics)
    print('')

    plot_types = viz_cfg.get('plot_types')
    plot_types_dict = {k: v for d in plot_types for k, v in d.items()}
    true_keys = [key for key, value in plot_types_dict.items() if value is True]
    print('The following plots will be generated:')
    print(true_keys)
    print('')

    # Get features from the pred config file --------------------------
    path_pred_config = fsate.build_cfig_path(path_viz_config,viz_cfg.get('name_pred_config',None)) # currently, this gives the pred config path, not the attr config path
    pred_cfg = yaml.safe_load(open(path_pred_config, 'r'))
    path_attr_config = fsate.build_cfig_path(path_pred_config,pred_cfg.get('name_attr_config',None)) 

    # Get features from the attr config file --------------------------
    with open(path_attr_config, 'r') as file:
        attr_cfg = yaml.safe_load(file)

    datasets = list([x for x in attr_cfg['formulation_metadata'] if 'datasets' in x][0].values())[0] # Identify datasets of interest
    dir_base = list([x for x in attr_cfg['file_io'] if 'dir_base' in x][0].values())[0]
    dir_std_base = list([x for x in attr_cfg['file_io'] if 'dir_std_base' in x][0].values())[0]
    dir_std_base = f'{dir_std_base}'.format(dir_base = dir_base)
    # Options for getting ds_type from a config file:
    # ds_type = viz_cfg.get('ds_type') # prediction config file IF VISUALIZING PREDICTIONS; attribute config file IF AND ONLY IF VISUALIZING ATTRIBUTES
    # ds_type = list([x for x in attr_cfg['file_io'] if 'ds_type' in x][0].values())[0]
    # ...but for plotting purposes, we want to use the prediction ds_type:
    ds_type = 'prediction'
    write_type = list([x for x in attr_cfg['file_io'] if 'write_type' in x][0].values())[0]

    # Get features from the main config file --------------------------
    # NOTE: This assumes that the main config file is just called [same prefix as all other config files]_config.yaml
    prefix_viz = str(path_viz_config.name).split('_')[0]
    prefix_attr = str(path_attr_config.name).split('_')[0]
    if (prefix_viz != prefix_attr):
        raise ValueError('The base config file (e.g. [dataset]_config.yaml) must be in the same direcotry and identifiable using the same prefix as the other config files (e.g. [dataset]_pred_config.yaml, [dataset]_attr_config.yaml, etc.)')
    else:
        prefix = prefix_viz

    path_main_config = fsate.build_cfig_path(path_viz_config,f'{prefix_viz}_config.yaml')
    with open(path_main_config, 'r') as file:
        main_cfg = yaml.safe_load(file)

    # NOTE: This is something I'm not totally sure will function properly with multiple datasets
    formulation_id = list([x for x in main_cfg['formulation_metadata'] if 'formulation_id' in x][0].values())[0]
    save_type = list([x for x in main_cfg['file_io'] if 'save_type' in x][0].values())[0]
    if save_type.lower() == 'netcdf':
        save_type_obs = 'nc'
        engine = 'netcdf4'
    else:
        save_type_obs = 'zarr'
        engine = 'zarr'

    # Access the location metadata for prediction sites
    path_meta_pred = pred_cfg.get('path_meta')

    # Location for accessing existing outputs and saving plots
    dir_out = fsate.fs_save_algo_dir_struct(dir_base).get('dir_out')

    # Loop through all datasets
    for ds in datasets:
        path_meta_pred = f'{path_meta_pred}'.format(ds = ds, dir_std_base = dir_std_base, ds_type = ds_type, write_type = write_type)
        meta_pred = pd.read_parquet(path_meta_pred)

        # Loop through all algorithms
        for algo in algos:
            # Loop through all metrics
            for metric in metrics:
                # Pull the predictions
                path_pred = fsate.std_pred_path(dir_out,algo=algo,metric=metric,dataset_id=ds)
                pred = pd.read_parquet(path_pred)
                data = pd.merge(meta_pred, pred, how = 'inner', on = 'comid')
                os.makedirs(f'{dir_out}/data_visualizations', exist_ok= True)
                # If you want to export the merged data for any reason: 
                # data.to_csv(f'{dir_out}/data_visualizations/{ds}_{algo}_{metric}_data.csv')

                # Does the user want a scatter plot comparing the observed module performance and the predicted module performance by RaFTS?
                if 'perf_map' in true_keys:
                    states = gpd.read_file('/Users/laurenbolotin/data/conus_states_census.shp')
                    states = states.to_crs("EPSG:4326")

                    # Plot performance on map
                    lat = data['Y']
                    lon = data['X']
                    geometry = [Point(xy) for xy in zip(lon,lat)]
                    geo_df = gpd.GeoDataFrame(geometry = geometry)
                    geo_df['performance'] = data['prediction'].values
                    geo_df.crs = ("EPSG:4326")

                    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
                    base = states.boundary.plot(ax=ax,color="#555555", linewidth=1)
                    # Points
                    geo_df.plot(column="performance", ax=ax, markersize=150, cmap='viridis', legend=False, zorder=2) # delete zorder to plot points behind states boundaries
                    # States
                    states.boundary.plot(ax=ax, color="#555555", linewidth=1, zorder=1)  # Plot states boundary again with lower zorder

                    cbar = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-0.41,vmax = 1), cmap='viridis')
                    ax.tick_params(axis='x', labelsize= 24)
                    ax.tick_params(axis='y', labelsize= 24)
                    plt.xlabel('Latitude',fontsize = 26)
                    plt.ylabel('Longitude',fontsize = 26)
                    cbar_ax = plt.colorbar(cbar, ax=ax,fraction=0.02, pad=0.04)
                    cbar_ax.set_label(label=metric,size=24)
                    cbar_ax.ax.tick_params(labelsize=24)  # Set colorbar tick labels size
                    plt.title("Predicted Performance: {}".format(ds), fontsize = 28)

                    # Save the plot as a .png file
                    output_path = f'{dir_out}/data_visualizations/{ds}_{algo}_{metric}_performance_map.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.clf()
                    plt.close()

                if 'obs_vs_sim_scatter' in true_keys:
                    # Scatter plot of observed vs. predicted module performance
                    # Remove 'USGS-' from ids so it can be merged with the actual performance data
                    data['identifier'] = data['identifier'].str.replace(r'\D', '', regex=True)
                    data['identifier'] = data['identifier'].str.strip() # remove leading and trailing spaces

                    # Read in the observed performance data
                    path_obs_perf = f'{dir_std_base}/{ds}/{ds}_{formulation_id}.{save_type_obs}'
                    obs = xr.open_dataset(path_obs_perf, engine=engine)
                    # NOTE: Below is one option, but it assumes there is only one possible .nc or .zarr file to read in (it only reads the first one it finds with that file extension)
                    # obs = fsate._open_response_data_fs(dir_std_base=dir_std_base, ds=ds)
                    obs = obs.to_dataframe()

                    # Standardize column names
                    obs.reset_index(inplace=True)
                    obs = obs.rename(columns={"gage_id": "identifier"})

                    # Subset columns
                    data = data[['identifier', 'comid', 'X', 'Y', 'prediction', 'metric', 'dataset']]
                    data = data[data['metric'] == metric]
                    data.columns = data.columns.str.lower()
                    obs = obs[['identifier', metric]]

                    # Merge the observed and predicted data
                    data = pd.merge(data, obs, how = 'inner', on = 'identifier')

                    # Plot the observed vs. predicted module performance
                    plt.scatter(data['prediction'], data[metric], c='teal')
                    plt.axline((0, 0), (1, 1), color='black', linestyle='--')
                    plt.xlabel('Predicted {}'.format(metric))
                    plt.ylabel('Actual {}'.format(metric))
                    plt.title('Observed vs. Predicted Performance: {}'.format(ds))

                    # Save the plot as a .png file
                    output_path = f'{dir_out}/data_visualizations/{ds}_{algo}_{metric}_obs_vs_sim_scatter.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                



'''ARCHIVE CODE'''
## This is how I was pulling info from the data viz config file before I figured out a better way to make file paths
''' path_cfg_dir = viz_cfg.get('cfg_dir')
ds = viz_cfg.get('ds')
dir_base = viz_cfg.get('dir_base')
dir_std_base = viz_cfg.get('dir_std_base')
ds_type = viz_cfg.get('ds_type')
write_type = viz_cfg.get('write_type')

path_cfg_dir = f'{path_cfg_dir}/'.format(ds =ds, dir_base = dir_base)
cfg_yamls = os.listdir(path_cfg_dir)

# Pull prediction config yaml
pred_cfg_str = 'pred_config.yaml'
path_pred_cfg = [element for element in cfg_yamls if pred_cfg_str in element and ds in element]
if len(path_pred_cfg) == 0:
    raise ValueError(f"Ensure that 'pred_config.yaml' is in the directory {path_cfg_dir}")
if len(path_pred_cfg) > 1:
    raise ValueError(f"Multiple 'pred_config.yaml' files found in the directory {path_cfg_dir}")
path_pred_cfg = f'{path_cfg_dir}{path_pred_cfg[0]}'
pred_cfg = yaml.safe_load(open(path_pred_cfg, 'r'))

# Access the location metadata for prediction sites
path_meta_pred = pred_cfg.get('path_meta')
path_meta_pred = f'{path_meta_pred}/'.format(ds =ds, dir_std_base = dir_std_base, ds_type = ds_type, write_type = write_type, dir_base = dir_base)
meta_pred = pd.read_parquet(path_meta_pred)

print(path_meta_pred)'''

## This is how you would (potentially) get path_meta from the attr config
# Pull attribute config yaml
# attr_cfg_str = 'attr_config.yaml'
# path_attr_cfg = [element for element in cfg_yamls if attr_cfg_str in element and ds in element]
# if len(path_attr_cfg) == 0:
#     raise ValueError(f"Ensure that 'attr_config.yaml' is in the directory {path_cfg_dir}")
# if len(path_attr_cfg) > 1:
#     raise ValueError(f"Multiple 'attr_config.yaml' files found in the directory {path_cfg_dir}")
# path_attr_cfg = f'{path_cfg_dir}{path_attr_cfg[0]}'
# print(path_attr_cfg)

# attr_cfg = fsate.AttrConfigAndVars(path_attr_cfg)
# attr_cfg._read_attr_config()
# dir_base = attr_cfg.attrs_cfg_dict.get('dir_base')
# print(dir_base)
# path_meta = attr_cfg.attrs_cfg_dict.get('path_meta')
# print(path_meta)

## But I believe we only need the one from the pred config
# 
# ----------------------------------- 
# print(os.listdir(path_cfg_dir))
# # attr_cfg = fsate.AttrConfigAndVars(path_attr_config)

# ----------------------------------------------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description = 'process the data visualization config file')
#     parser.add_argument('path_viz_config', type=str, help='Path to the YAML configuration file specific for data visualization')
#     args = parser.parse_args()

#     home_dir = Path.home()
#     path_viz_config = Path(args.path_viz_config) #Path(f'{home_dir}/FSDS/formulation-selector/scripts/eval_ingest/xssa/xssa_viz_config.yaml') 

#     with open(path_viz_config, 'r') as file:
#         viz_cfg = yaml.safe_load(file)
#     # print(viz_cfg['cfg_dir'])

#     # Get features from the data viz config file
#     path_cfg_dir = viz_cfg.get('cfg_dir')
#     ds = viz_cfg.get('ds')
#     dir_base = viz_cfg.get('dir_base')
#     dir_std_base = viz_cfg.get('dir_std_base')
#     dir_std_base = f'{dir_std_base}'.format(dir_base=dir_base)
#     dir_cfg_base = viz_cfg.get('dir_cfg_base')
#     ds_type = viz_cfg.get('ds_type')
#     write_type = viz_cfg.get('write_type')

#     path_cfg_dir = f'{path_cfg_dir}/'.format(ds =ds, dir_cfg_base = dir_cfg_base)
#     cfg_yamls = os.listdir(path_cfg_dir)

#     # Pull prediction config yaml
#     pred_cfg_str = 'pred_config.yaml'
#     path_pred_cfg = [element for element in cfg_yamls if pred_cfg_str in element and ds in element]
#     if len(path_pred_cfg) == 0:
#         raise ValueError(f"Ensure that 'pred_config.yaml' is in the directory {path_cfg_dir}")
#     if len(path_pred_cfg) > 1:
#         raise ValueError(f"Multiple 'pred_config.yaml' files found in the directory {path_cfg_dir}")
#     path_pred_cfg = f'{path_cfg_dir}{path_pred_cfg[0]}'
#     pred_cfg = yaml.safe_load(open(path_pred_cfg, 'r'))

#     # Access the location metadata for prediction sites
#     path_meta_pred = pred_cfg.get('path_meta')
#     print(path_meta_pred)
#     path_meta_pred = f'{path_meta_pred}/'.format(ds =ds, dir_std_base = dir_std_base, ds_type = ds_type, write_type = write_type)
#     print(path_meta_pred)

#     meta_pred = pd.read_parquet(path_meta_pred)
#     # FileNotFoundError: [Errno 2] No such file or directory: '/Users/laurenbolotin/Lauren/FSDS/data/input/user_data_std/xssa/nldi_feat_xssa_prediction.parquet/'
#     # It can't find the file because the file has a longer dataset name (juliemai-xssa) than the one in the config file (xssa)
#     # Need to resolve this,  maybe by extracting the dataset name from the attribute config file instead of the viz config file
#     # But I also want to understand why there are two different dataset names to begin with

## Config file contents that worked with this: 
# dir_base: '/Users/laurenbolotin/Lauren/FSDS/data/input' # Required. The save location of standardized output
# dir_std_base: '{dir_base}/user_data_std'
# dir_cfg_base: '/Users/laurenbolotin/Lauren/FSDS/eval_ingest_lb_28' 
# ds_type: 'prediction' # Required string. Recommended to select 'training' or 'prediction', but any string will work. This string will be used in the filename of the output metadata describing each data point's identifer, COMID, lat/lon, reach name of the location. This string should differ from the string used in the prediction config yaml file. Filename: `"nldi_feat_{dataset}_{ds_type}.csv"` inside `dir_std_base / dataset / `
# write_type: 'parquet'
# cfg_dir: '{dir_cfg_base}/{ds}' # Required. The directory where the config files are stored. The {ds} is the dataset name.
# ds: 'xssa' # Required. The dataset name.

# When i was pulling the attr config instead of the pred config and THEN the attr config:
    # path_pred_config = Path('/Users/laurenbolotin/Lauren/FSDS/eval_ingest_lb_28/xssa/xssa_pred_config.yaml') #Path(f'{home_dir}/git/formulation-selector/scripts/eval_ingest/xssa/xssa_pred_config.yaml') 
    # with open(path_pred_config, 'r') as file:
    #     pred_cfg = yaml.safe_load(file)
    # path_attr_config = fsate.build_cfig_path(path_pred_config,pred_cfg.get('name_attr_config',None))
    # print('pring the path_attr_config that was generated by the build_cfig_path function')
    # print(path_attr_config)


    # attr_cfig = fsate.AttrConfigAndVars(path_attr_config)
    # print(attr_cfig)
    # attr_cfig._read_attr_config()
    # datasets = attr_cfig.attrs_cfg_dict.get('datasets') # Identify datasets of interest
    # print(datasets)


# From when I was trying to get a list of config files by having the user specify a config directory rather than having it deduced from just the input arg (path_viz_config)

    # Get features from the data viz config file
    # path_cfg_dir = viz_cfg.get('cfg_dir')
    # # NOTE: I think dir_cfg_base is the only one that would actually need a new value from a new config file
    # dir_cfg_base = viz_cfg.get('dir_cfg_base')
    # path_cfg_dir = f'{path_cfg_dir}/'.format(ds = ds, dir_cfg_base = dir_cfg_base)
    # cfg_yamls = os.listdir(path_cfg_dir)

    # # TODO: I think ds,  dir base, ds, write_type, and dir_std_base can come from another config file
    # # Pull attribute config yaml
    # attr_cfg_str = 'attr_config.yaml'
    # path_attr_cfg = [element for element in cfg_yamls if attr_cfg_str in element]
    # if len(path_attr_cfg) == 0:
    #     raise ValueError(f"Ensure that 'attr_config.yaml' is in the directory {path_cfg_dir}")
    # if len(path_attr_cfg) > 1:
    #     raise ValueError(f"Multiple 'attr_config.yaml' files found in the directory {path_cfg_dir}")
    # path_attr_cfg = f'{path_cfg_dir}{path_attr_cfg[0]}'
    # attr_cfg = fsate.AttrConfigAndVars(path_attr_cfg)
    # print(attr_cfg)

    # ds = viz_cfg.get('ds') # attribute config file
    # dir_base = viz_cfg.get('dir_base') # attribute config file
    # dir_std_base = viz_cfg.get('dir_std_base') # attribute config file
    # dir_std_base = f'{dir_std_base}'.format(dir_base = dir_base)
    # ds_type = viz_cfg.get('ds_type') # prediction config file IF VISUALIZING PREDICTIONS; attribute config file IF AND ONLY IF VISUALIZING ATTRIBUTES
    # write_type = viz_cfg.get('write_type') # either prediction or attribute config file, whichever is easier, or keep it consistent with the line above

    # # Pull prediction config yaml
    # pred_cfg_str = 'pred_config.yaml'
    # # path_pred_cfg = [element for element in cfg_yamls if pred_cfg_str in element and ds in element] # Previous code for also looking for the ds name in the config filenames
    # path_pred_cfg = [element for element in cfg_yamls if pred_cfg_str in element]
    # if len(path_pred_cfg) == 0:
    #     raise ValueError(f"Ensure that 'pred_config.yaml' is in the directory {path_cfg_dir}")
    # if len(path_pred_cfg) > 1:
    #     raise ValueError(f"Multiple 'pred_config.yaml' files found in the directory {path_cfg_dir}")
    # path_pred_cfg = f'{path_cfg_dir}{path_pred_cfg[0]}'
    # pred_cfg = yaml.safe_load(open(path_pred_cfg, 'r'))

    # # Access the location metadata for prediction sites
    # path_meta_pred = pred_cfg.get('path_meta')
    # path_meta_pred = f'{path_meta_pred}/'.format(ds = ds, dir_std_base = dir_std_base, ds_type = ds_type, write_type = write_type)

    # meta_pred = pd.read_parquet(path_meta_pred)
    # FileNotFoundError: [Errno 2] No such file or directory: '/Users/laurenbolotin/Lauren/FSDS/data/input/user_data_std/xssa/nldi_feat_xssa_prediction.parquet/'
    # It can't find the file because the file has a longer dataset name (juliemai-xssa) than the one in the config file (xssa)
    # Need to resolve this,  maybe by extracting the dataset name from the attribute config file instead of the viz config file
    # But I also want to understand why there are two different dataset names to begin with

# This is from when a bunch of details were coming from the data viz config file (which duplicated information) rather than getting it from config files that already had it: 
# ds = viz_cfg.get('ds') # attribute config file
# dir_base = viz_cfg.get('dir_base') # attribute config file
# dir_std_base = viz_cfg.get('dir_std_base') # attribute config file
# dir_std_base = f'{dir_std_base}'.format(dir_base = dir_base)
# ds_type = viz_cfg.get('ds_type') # prediction config file IF VISUALIZING PREDICTIONS; attribute config file IF AND ONLY IF VISUALIZING ATTRIBUTES
# write_type = viz_cfg.get('write_type') # either prediction or attribute config file, whichever is easier, or keep it consistent with the line above

# From when I was using the functions specific to the attr config file for reading its contents rather than the more generic [config].get() method:
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description = 'process the data visualization config file')
#     parser.add_argument('path_viz_config', type=str, help='Path to the YAML configuration file specific for data visualization')
#     args = parser.parse_args()

#     home_dir = Path.home()
#     path_viz_config = Path(args.path_viz_config) #Path(f'{home_dir}/FSDS/formulation-selector/scripts/eval_ingest/xssa/xssa_viz_config.yaml') 

#     with open(path_viz_config, 'r') as file:
#         viz_cfg = yaml.safe_load(file)
#     # print(viz_cfg['cfg_dir'])

#     # Get features from the pred config file
#     path_pred_config = fsate.build_cfig_path(path_viz_config,viz_cfg.get('name_pred_config',None)) # currently, this gives the pred config path, not the attr config path
#     pred_cfg = yaml.safe_load(open(path_pred_config, 'r'))
#     path_attr_config = fsate.build_cfig_path(path_pred_config,pred_cfg.get('name_attr_config',None)) 

#     # Get features from the attr config file
#     attr_cfg = fsate.AttrConfigAndVars(path_attr_config) 
#     attr_cfg._read_attr_config()
#     datasets = attr_cfg.attrs_cfg_dict.get('datasets') # Identify datasets of interest

#     dir_base = attr_cfg.attrs_cfg_dict.get('dir_base')
#     print('dir_base:')
#     print(dir_base)

#     dir_std_base = attr_cfg.attrs_cfg_dict.get('dir_std_base')

#     # ds_type = viz_cfg.get('ds_type') # prediction config file IF VISUALIZING PREDICTIONS; attribute config file IF AND ONLY IF VISUALIZING ATTRIBUTES
#     ds_type = attr_cfg.attrs_cfg_dict.get('ds_type')
#     print('ds_type:')
#     print(ds_type)

#     write_type = attr_cfg.attrs_cfg_dict.get('write_type')
#     print('write_type:')
#     print(write_type)

#     # Access the location metadata for prediction sites
#     path_meta_pred = pred_cfg.get('path_meta')
#     # TODO: don't have it assume you're only working with one dataset (datasets[0]), this is where a loop will come in
#     for ds in datasets:
#         path_meta_pred = f'{path_meta_pred}'.format(ds = ds, dir_std_base = dir_std_base, ds_type = ds_type, write_type = write_type)
#         print('path_meta_pred')
#         print(path_meta_pred)

#         meta_pred = pd.read_parquet(path_meta_pred)