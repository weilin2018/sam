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
import cartopy.crs as ccrs
import cartopy as ctp
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../../") # Adds higher directory 

from src.dataset import BaseDataset, load_dataset
from src.climnet import ClimNet

#%%
"""Create network of Event series calculation of precipitation."""


class Asia_Network(BaseDataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str  
        filename  
    var_name: str
        Variable name of interest
    """

    def __init__(self, nc_file, var_name=None,
                 lon_range=[-180, 180], lat_range=[-90, 90],
                 time_range=['1998-01-01', '2019-01-01'],
                 grid_step=1.0, anomalies=False, name=None, lsm=False, rrevs=False, load=False):
            
        super().__init__(nc_file, var_name,
                lon_range=lon_range, lat_range=lat_range,
                time_range=time_range, grid_step=grid_step,
                name=name,
                anomalies=anomalies, lsm=lsm, evs=True, rrevs=rrevs,load=load)
        
    
    def create_es_network(self,  name, grid_step, run_evs=False, linkbund=True, savenet=True, num_jobs=1,
                          num_cpus=16):
        print('Create and store Network')
        Network = ClimNet(self, corr_method='es', num_jobs=num_jobs, run_evs=True  )

        # store 
        if savenet == True:
            networkfile = PATH + f"/../outputs/{name}_ES_net_{grid_step}.npz"
            network = Network.convert2sparse(Network.adjacency)
            Network.store_network(network, networkfile)

        # link bundles
        if linkbund == True:
            print('Start linkbundles...')
            adjacency_corr = Network.link_bundles(
                adjacency=Network.adjacency,
                confidence=0.99, num_rand_permutations=1000,
                num_cpus=num_cpus
            )
            
            if savenet == True:
                networkfile = PATH + f"/../outputs/{name}_link_bundle_net_{grid_step}.npz"
                network_corr = Network.convert2sparse(adjacency_corr)
                Network.store_network(network_corr, networkfile)
        
        return Network

    def create_empty_Network(self):
        print('Create and store Network Instance')
        Network = ClimNet(self,  corr_method='es',run_evs=False  ) 
        return Network
# %%    
if __name__ == "__main__":
    
    trmm=True

    load_ds=True
    es_run=False
    
    load_net = False

    grid_step = 0.5
    vname = 'pr'
    anomalies=True
    
    name='gpcp_asia_es'   
    if trmm:
        name='trmm_asia_es'
        grid_step = 0.5


    num_cpus = 64
    num_jobs=30
    time_range=['1998-01-01', '2019-01-01']
    
    # Asia subcontinent range
    lat_range=[-15, 45]
    lon_range=[55, 150]
    if os.getenv("HOME") =='/home/goswami/fstrnad80':
        dirname = "/home/goswami/fstrnad80/data/GPCP/"
        if trmm is True:
            dirname = "/home/goswami/fstrnad80/data/trmm/"
            time_range=['1998-01-01', '2019-01-01']


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
    # %%
    print('Loading Data')
    dataset_file = PATH + f"/../../outputs/{name}_ds_{grid_step}.nc"
    if anomalies is True:
        dataset_file = PATH + f"/../../outputs/{name}_ds_{grid_step}_anomalies.nc"

    if load_ds is True:
        ds = Asia_Network(dataset_file, load=load_ds, rrevs=True, name=name)

    else:
        print('Create Dataset')
        ds = Asia_Network(fname, var_name=vname, name=name,
                        lon_range=lon_range, lat_range=lat_range,
                        time_range=time_range,
                        grid_step=grid_step, 
                        anomalies=anomalies)
        ds.save(dataset_file)
        sys.exit(0)

    # %%
    # create network
    if load_net is False:
        PrecipNet = ds.create_empty_Network()
        null_model_file=PATH + '/../null_model/pr/null_model_pr_mnoe_402_tau_max_10_threshold_001.npy'
        networkfile = PATH + f"/../../outputs/{name}_ES_net_{grid_step}.npz"
        networkfile_lb = PATH + f"/../../outputs/{name}_ES_net_lb_{grid_step}.npz"

        PrecipNet.adjacency=PrecipNet.full_event_synchronization_run(null_model_file=null_model_file, networkfile=networkfile,
                                        num_jobs=num_jobs, num_cpus=num_cpus, es_run=es_run, c_adj=False, linkbund=True )
        sys.exit(0)
    else:
        # load network
        networkfile = PATH + f"/../../outputs/{name}_ES_net_{grid_step}.npz"
        PrecipNet = ClimNet(ds, network_file=networkfile)

        networkfile_lb = PATH + f"/../../outputs/{name}_ES_net_lb_{grid_step}.npz"
        PrecipNet_lb = ClimNet(ds, network_file=networkfile_lb)
    
    # %%
    # Plot Q values 
    
    q_mean = ds.mean_val
    q_median=ds.q_median
    q_map = ds.q_mask
    q_rel_frac=ds.rel_frac_q_map
    # %%
    ds.plot_map(q_mean, color='coolwarm_r', label=f'Mean Rainfall [mm/days]', projection='PlateCarree',
                vmax=30, plt_mask=False)
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_mean.png")

    # %%
    ds.plot_map(q_map, color='coolwarm_r', label=f'Quantile {ds.q} values [mm/days]', projection='PlateCarree', 
                vmax=110)
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_q_mask.pdf")

    # %%
    ds.plot_map(q_median, color='coolwarm_r', label=f'Median Rainfall [mm/days]', projection='PlateCarree')
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_median.pdf")

    num_evs=ds.num_eev_map
    ds.plot_map(num_evs, color='RdBu', label=f'Number of ERE (q={ds.q})', 
                vmax=300,projection='PlateCarree', 
                )
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_num_ERE.pdf")  

    # %%
    ds.plot_map(q_rel_frac, color='hot', label=f'Relative Fraction ERE', projection='PlateCarree',
                plt_mask=False)
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_rel_fraction_ERE.png", 
                )

    # %%
    
    fig, ax= plt.subplots(figsize=(12,6), nrows=1, ncols=2, subplot_kw={'projection': ccrs.PlateCarree()})
    ds.plot_map(q_mean, color='coolwarm_r', label=f'Mean Rainfall [mm/days]', projection='PlateCarree',
                plt_mask=False, vmax=30, ax=ax[0], fig=fig)
    ds.plot_map(q_map, color='coolwarm_r', label=f'Quantile {ds.q} values [mm/days]', plt_mask=False, 
                projection='PlateCarree', vmax=120, ax=ax[1], fig=fig)
    # plt.tight_layout()
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_mean_quantile.png")
    

# %% 
    # Plot  mask
    mask=ds.mask
    ds.plot_map(mask, color='Blues', label=f'Definition values', projection='PlateCarree')
    plt.savefig(PATH + f"/../../plots/{name}_grid_{grid_step}_ES_mask.pdf")

    # %%
    # Plot links
    lon_range=[98,102]
    lat_range=[12,16]
    loc=(np.mean(lon_range), np.mean(lat_range))
    def get_link_idx_lst(self, idx_lst):
        
        link_lst=[]
        for idx in idx_lst:
            links=list(np.where(self.adjacency[idx,:]==1)[0])
            link_lst=list(set( link_lst + links ))

        return link_lst

    indices = ds.get_idx_range(lon_range=lon_range, lat_range=lat_range)
    link_list=get_link_idx_lst(PrecipNet_lb, indices)
    link_map = ds.get_map(ds.flat_idx_array( link_list) )
    ds.plot_contour_map(link_map, color='black', cmap='Reds_r', projection='PlateCarree', 
                        fill_out=False, n_contour=2, vmin=0, vmax=1, title=f'Network {loc} with link bundle', bar=False)
    plt.savefig(PATH + f"/../../plots/{name}_grid_{grid_step}_node_{loc}.pdf")
    
    # %%
    link_map = ds.get_map(PrecipNet.adjacency[idx,:])
    ds.plot_map(link_map, color='black', projection='PlateCarree', cmap='Reds_r',
                        fill_out=False, title=f'Network {loc} with no link bundle', bar=False, vmin=0, vmax=1)
    plt.savefig(PATH + f"/../../plots/{name}_grid_{grid_step}_node_{idx}_lb.pdf")

    # %%
    # Plot nodedegree
    fig = plt.figure(figsize=(15,10))
    nd_map = ds.get_map(PrecipNet.get_node_degree(), name=f'node_degree_{name}')
    ds.plot_map(nd_map, color='Reds', vmin=0, vmax=None, projection='PlateCarree', title='Node degree')
    plt.savefig(PATH + f"/../../plots/{name}_ES_grid_{grid_step}_nodedegree.pdf")

    # %%
    # Plot adjacency
    M, N = PrecipNet.adjacency.shape
    extent = [0, M, 0, N]
    fig = plt.figure(figsize=(15,10))
    plt.imshow(PrecipNet.adjacency[::-1, :], extent=extent)
    plt.savefig(PATH + f"/../../plots/{name}_grid_{grid_step}_adjacency.pdf")


    # %%
