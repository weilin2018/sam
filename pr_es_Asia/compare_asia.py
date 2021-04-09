#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
#%%
import sys, os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../../") # Adds higher directory 
from importlib import reload

from src.climnet import ClimNet
from src.dendrograms import Dendrogram_ES
from src.es_graph_tool import ES_Graph_tool
from src.compare_runs import Compare_Runs
from src.analyse_network import Analyse_Network
from src.monsoon_region_es import Monsoon_Region_ES
# %%
if __name__ == "__main__":
    trmm=True
    run_gt=True

    grid_step = 1
    vname = 'pr'
    
    name='asia_es'   
    if trmm:
        grid_step = 1
        name='trmm_asia_es'

    num_cpus = 64
    num_jobs=16
    
    # Asia subcontinent range
    lat_range=[-15, 45]
    lon_range=[55, 150]
    if os.getenv("HOME") =='/home/goswami/fstrnad80':
        dirname = "/home/goswami/fstrnad80/data/GPCP/"
        if trmm is True:
            dirname = "/home/goswami/fstrnad80/data/trmm/"

    else:
        dirname = "/home/strnad/data/GPCP/"
        if trmm is True:
            dirname = "/home/strnad/data/trmm/"

        # dirname = "/home/jakob/climate_data/local/"
        num_cpus = 16

    
    # fname = dirname +f'{name}_gpcp_daily_1996_2020_{grid_step}p.nc4'
    fname = dirname +f'gpcp_daily_1996_2020_new.nc4'
    if trmm is True:
        fname = dirname +f'trmm_pr_daily_1998_2019.nc4'

    if os.getenv("HOME") =='/home/goswami/fstrnad80':
        dirname = "/home/goswami/fstrnad80/data/GPCP/"
    else:
        dirname = "/home/strnad/data/GPCP/"
        # dirname = "/home/jakob/climate_data/local/"
        num_cpus = 16
    #for job array
    try:
        min_job_id=int(os.environ['SLURM_ARRAY_TASK_MIN'])
        max_job_id=int(os.environ['SLURM_ARRAY_TASK_MAX'])
        job_id  =  int(os.environ['SLURM_ARRAY_TASK_ID'])
        num_jobs=int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        # num_jobs=max_job_id-min_job_id +1
        # num_jobs=16

        print(f"job_id: {job_id}/{num_jobs}, Min Job ID: {min_job_id}, Max Job ID: {max_job_id}" )
    except KeyError:
        job_id=2

    
    networkfile_lb = PATH + f"/../../outputs/{name}_ES_net_lb_{grid_step}.npz"
    networkfile_lb = PATH + f"/../../outputs/{name}_ES_net_{grid_step}.npz"

    sbm_filepath=PATH + f"/../graphs/{name}_{grid_step}/{job_id}_{name}_graph_tool_ES_{grid_step}"    
    
    print('Loading Data')
    load_ds = True
    dataset_file = PATH + f"/../../outputs/{name}_ds_{grid_step}.nc"

    


    # %% 
    # Now compare runs
    num_runs=10
    sbm_filepath=PATH + f"/../graphs/{name}_{grid_step}/{job_id}_{name}_graph_tool_ES_{grid_step}"

    group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )    
    
    ds_cr = Analyse_Network(nc_file=dataset_file, name=name, 
                              network_file=networkfile_lb,
                              group_levels=group_levels,
                              load=True)
    # %%
    # Get specific monsoon paths
    m_arr=['South Asia', 'India ' ,  'East Asia', ]

    monsoon_dict=ds_cr.monsoon_dictionary
    sel_Z, node_ids=ds_cr.plot_monsoon_paths(m_arr=m_arr, title=f'Dendrogram for JobID # {job_id}') 
    savepath=PATH + f"/../../plots/asia/cluster_analysis/asia_monsoon_dendrogram.pdf"
    plt.savefig(savepath,  bbox_inches='tight')  
    
    
    # %%
    # Rep Ids visualization
    n_idx_list=[]
    for mname, mregion in monsoon_dict.items():
        n_idx_list.append(mregion['rep_ids'])
    loc_map = ds_cr.get_map(ds_cr.flat_idx_array(np.array(n_idx_list).flatten()))
    ax=ds_cr.plot_map(loc_map, color='Reds', title=None, bar=False, projection='PlateCarree')
    # ds_cr.plot_contour_map(ds_cr.ee_def, color='Blues', ax=ax, bar=False, fill_out=False, projection='PlateCarree')
    savepath=PATH + f"/../../plots/asia/cluster_analysis/asia_monsoon_regions.pdf"

    plt.savefig(savepath)


    # %% 
    savepath=PATH + f"/../../plots/asia/entropy_multiple_runs_asia.pdf"

    mean_entr, std_entr, mean_ngroups, std_ngroups= ds_cr.compare_entropy(num_runs=10,max_num_levels=7,
                                                                          savepath=savepath)
    # %%
    sel_m=['South India', 'South Asia']
    
    c_Zid, sel_mids= ds_cr.get_ms_cluster_ids(m_arr=sel_m)
    leaf_nodes= ds_cr.d_es.d_tree.get_leaf_nodes_of_is_node(c_Zid)
   
    lmap=ds_cr.get_map(ds_cr.flat_idx_array( leaf_nodes) )
    ds_cr.plot_map(lmap, color='Reds_r', projection='PlateCarree', 
                        vmin=0, vmax=1, title=f'Cluster for {sel_m}', bar=False)


    # %%
    import src.link_bundles as lb
    from tqdm import tqdm
    scott_factor=0.02
    savepath=PATH + f"/../../plots/asia/cluster_analysis/density_cluster_monsoons_{m_arr}.pdf"
    
    def compare_monsoon_arr(self, num_runs=10, m_arr=None,  scott_factor=1, plot=False, graph_file=None):
        if m_arr is None:
            raise ValueError(f"Please select array of monsoon keys of specific locations!")
        
        
        cluster_idx_arr=np.zeros((num_runs, len(self.indices_flat)))
        av_num_links=np.count_nonzero(self.climnet.adjacency)/self.climnet.adjacency.shape[0]
        # av_num_links=500
        bw_opt= scott_factor * av_num_links**(-1./(2+4)) # Scott's rule of thumb
        for idx, job_id in tqdm(enumerate(range(0,num_runs))):
            if graph_file is None:
                sbm_filepath=(self.PATH + 
                              f"/graphs/{self.dataset_name}_{self.grid_step}/{job_id}_{self.dataset_name}_graph_tool_ES_{self.grid_step}")
            else:
                sbm_filepath=self.PATH + graph_file

            if not os.path.exists(sbm_filepath +'_group_levels.npy'):
                print(f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
                continue
            group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )
            d_es=Dendrogram_ES(group_levels,)

            c_Zid, mids =self.get_ms_cluster_ids(d_es=d_es, m_arr=m_arr, key_ids='rep_ids', abs_th=4, rel_th=0)
            if plot is True:
                self.plot_Z_id(Z_id= c_Zid, d_es=d_es, title=f'JobID: {job_id}')

            leaf_nodes= d_es.d_tree.get_leaf_nodes_of_is_node(c_Zid)
            
            if len(leaf_nodes) == len(self.indices_flat):
                print(f"JobID {job_id}: Warning all nodes in one cluster!")
            
            for mid in mids:
                if mid not in leaf_nodes:
                    print(f"JobId {job_id}: Warning monsoon id {mid} not in cluster ids!")

            cluster_idx_arr[idx,:]=self.flat_idx_array(leaf_nodes)
            
        mean=np.mean(cluster_idx_arr, axis=0)
        std=np.std(cluster_idx_arr, axis=0)

        return mean, std, m_arr


    def get_main_loc_gm(self, m_arr, num_runs=30, den_th=0, scott_factor=0.005, plot=False, 
                        projection='Mollweide',savepath=None, ax=None ):
        
        mean_cluster, std_cluster, m_arr=compare_monsoon_arr(self, num_runs=num_runs, m_arr=m_arr, plot=False, 
                                    scott_factor=scott_factor,)

        eps=0.2
        mean_cluster=np.where(mean_cluster>eps, mean_cluster, np.nan)
        mi_indices=np.where(mean_cluster > den_th)[0]
        if plot is True:
            main_monsoon_map=self.get_map(self.flat_idx_array(mi_indices))
            
            ax=self.plot_map(self.get_map(mean_cluster), ax=ax, label='Density of occurence',
                    title=f'Density Plot for {m_arr}', color='Reds',
                vmin=0, vmax=None, projection=projection )
            self.plot_contour_map(main_monsoon_map, ax=ax, fill_out=False,
                        cmap='Greys', vmin=0, vmax=1, projection=projection )
            if savepath is not None:
                plt.savefig(savepath)
        
        return mi_indices

    mi_indices=get_main_loc_gm(ds_cr, sel_m, den_th=.90, num_runs=30, scott_factor=scott_factor, 
                                plot=True,savepath=savepath, projection='PlateCarree' )

    # %%

    import cartopy.crs as ccrs
    import cartopy as ctp

    projection=ccrs.PlateCarree()
    scott_factor=0.001

    fig, ax= plt.subplots(figsize=(12,6), nrows=1, ncols=2, subplot_kw={'projection': projection})
    m_arr=['East Asia', 'CWP']
    mi_indices=ds_cr.get_main_loc_gm(m_arr, den_th=.7, num_runs=10, scott_factor=scott_factor, 
                                plot=True,savepath=None, projection='PlateCarree', ax=ax[0] )
    m_arr=['India', 'South Asia']
    mi_indices=ds_cr.get_main_loc_gm(m_arr, den_th=.7, num_runs=10, scott_factor=scott_factor, 
                                plot=True,savepath=None, projection='PlateCarree', ax=ax[1] )

    savepath=PATH + f"/../../plots/asia/cluster_analysis/density_cluster_monsoons_North_South.pdf"
    plt.savefig(savepath)

    # %%
    mean_cluster, std_cluster, m_arr=ds_cr.compare_monsoon_arr(num_runs=30, m_arr=m_arr, plot=False, 
                                scott_factor=scott_factor)

    ds_cr.plot_map(ds_cr.get_map(std_cluster), label='Std of occurence', title=f'Density Plot for {m_arr}', color='Reds',
                vmin=0, vmax=None, projection='PlateCarree')
    plt.savefig(savepath)

    # %%
    scott_factor=0.005
    loc=(100, 0)
    num_last_levels=6
    mean_cluster, std_cluster=ds_cr.compare_grouping(num_runs=10, num_last_levels=num_last_levels,
                        loc=loc, scott_factor=scott_factor)
    # %%
    sel_level=2
    lid=len(group_levels)-sel_level
    savepath=PATH + f"/../../plots/asia/cluster_analysis/density_cluster_{lid}_monsoons.pdf"
    ds_cr.plot_map(ds_cr.get_map(mean_cluster[-sel_level]), label='Mean of occurence density', 
                title=f'Density Plot for cluster at {loc} at level {lid}', 
                color='Reds',
                vmin=0, vmax=None, projection='PlateCarree')
    # plt.savefig(savepath)
    # %%
    np.save(f'cluster_loc_{loc}.npy', mean_cluster)

    