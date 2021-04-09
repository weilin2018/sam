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

from src.pr_es_Asia.create_asia_network import Asia_Network
from src.climnet import ClimNet

#%%
"""Create network of Event series calculation of precipitation."""

# %%    
if __name__ == "__main__":
    
    trmm=True

    load_ds=True
    es_run=False
    
    load_net = True

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
    ee_map=ds.num_eev_map
    # %%
    ds.plot_map(q_mean, color='coolwarm_r', label=f'Mean Rainfall [mm/days]', projection='PlateCarree',
                vmax=30, plt_mask=False)
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_mean.png")

    # %%
    ds.plot_map(ee_map, color='coolwarm_r', label=f'Number of ERE', projection='PlateCarree',
                vmax=200, num_ticks=14, dcbar=True,  plt_mask=True)
    plt.savefig(PATH + f"/../../plots/asia/ee_plots/{name}_grid_{grid_step}_num_ee.png")

    # %%
    ds.plot_map(q_map, color='coolwarm_r', label=f'Quantile {ds.q} [mm/days]', projection='PlateCarree', 
                vmax=70, num_ticks=14, dcbar=True,  plt_mask=True)
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_quantile.pdf")


    # %%
    ds.plot_map(q_rel_frac, color='hot', label=f'Relative Fraction ERE', projection='PlateCarree',
                plt_mask=False)
    plt.savefig(PATH + f"/../../plots/asia/{name}_grid_{grid_step}_rel_fraction_ERE.png", 
                )

    # %% 
    # Plot together EEs and Q-values
    import string
    
    fig, ax= plt.subplots(figsize=(12,6), nrows=1, ncols=2, subplot_kw={'projection': ccrs.PlateCarree()})
    ds.plot_map(q_map, color='coolwarm_r', label=f'Quantile {ds.q} [mm/days]', projection='PlateCarree', 
                vmax=70, num_ticks=10, dcbar=True,  plt_mask=True, ax=ax[0], fig=fig)
    ds.plot_map(ee_map, color='coolwarm_r', label=f'Number of ERE', projection='PlateCarree',
                vmax=200, num_ticks=10, dcbar=True,  plt_mask=True, extend='max', ax=ax[1], fig=fig)
    for n, thisax in enumerate(ax):
        thisax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=thisax.transAxes, 
                size=20, weight='bold')
    plt.savefig(PATH + f"/../../plots/asia/ee_plots/{name}_grid_{grid_step}_ee_and_quantile.pdf", 
                bbox_inches = 'tight')
    

    # %% 
    # Get progression of number of ERE per Month
    ncols=2
    nrows= 3
    projection=ccrs.PlateCarree()
    savepath=PATH + f"/../../plots/asia/ee_plots/{name}_pr_ee_plots_season.png"


    fig,ax=plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols,6*nrows),
                        subplot_kw=dict(projection=projection ) )
    for lidx, month in enumerate(['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']):
        i = int(lidx/ncols)
        j= lidx-ncols*i
        this_ax=ax[i][j]
        ee_data=ds.get_month_range_data(ds.data_evs, start_month=month, end_month=month).sum(dim='time')
        ds.plot_map(ee_data, ax=this_ax, color='coolwarm_r', label=f'Number of EREs', title=month,
                    projection='PlateCarree', vmax=40, num_ticks=10, dcbar=True, plt_mask=True, extend='neither'
                    )
        this_ax.text(-0.1, 1.1, string.ascii_uppercase[lidx], transform=this_ax.transAxes, 
                size=20, weight='bold')
    plt.savefig(savepath, bbox_inches='tight')

    # viridis_r
    