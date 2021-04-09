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
from xarray.core.dataset import Dataset
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../../") # Adds higher directory 
from importlib import reload

from src.es_graph_tool import ES_Graph_tool
from src.compare_runs import Compare_Runs
from src.analyse_network import Analyse_Network
from src.monsoon_region_es import Monsoon_Region_ES
from src.dataset import BaseDataset
from src.pr_es_Asia.wind_presure_dataset import Wind_dataset



def plot_wind_field_ax(ax, u, v, steps=4, scale=50, width=0.005, headwidth=2.5):
    longs=u.coords['lon']
    lats=u.coords['lat']
    latsteps=lonsteps=steps
    u_dat= u.data[::latsteps, ::lonsteps]
    v_dat= v.data[::latsteps, ::lonsteps]
    
    ax.quiver(longs[::lonsteps], lats[::latsteps], u=u_dat, v=v_dat,
                            pivot='middle', transform=ccrs.PlateCarree(),
                            scale=scale, width=width, headwidth=headwidth)
    return ax
def plot_day_progression(ds, da, ds_wind, times, tps, re_map1=None, re_map2=None,  vmin=-5, vmax=5, label=None, savepath=None):
    ncols=2
    nrows= 4
    projection=ccrs.PlateCarree()
    import string
    fig,ax=plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols,5*nrows),
                        subplot_kw=dict(projection=projection ) )
    
    day_offset=-2
    for lidx, day in enumerate(range(day_offset,ncols*nrows + day_offset)):
        i = int(lidx/ncols)
        j= lidx-ncols*i
        this_ax=ax[i][j]

        title=f"Day {day}"
        
        an_map=da.sel(time=times[tps+day], method='nearest').mean(dim='time')
        ds.plot_map(an_map, ax=this_ax, title=title,label=label, plt_mask=False,
                    projection='PlateCarree', color='coolwarm_r', vmin=vmin, vmax=vmax)

        u=ds_wind.ds_uwind.data_anomalies.sel(time=times[tps+day], method='nearest').mean(dim='time') 
        v=ds_wind.ds_vwind.data_anomalies.sel(time=times[tps+day], method='nearest').mean(dim='time')
        this_ax=plot_wind_field_ax(ax=this_ax, u=u, v=v, )

        if re_map1 is not None:
            ds.plot_contour_map(re_map1, color='black',  projection='PlateCarree', 
                            fill_out=False, n_contour=2, vmin=0, vmax=1, bar=False, ax=this_ax)
        if re_map2 is not None:
            ds.plot_contour_map(re_map2, color='black', projection='PlateCarree', 
                            fill_out=False, n_contour=2, vmin=0, vmax=1, bar=False, ax=this_ax)
    
    this_ax.text(-0.1, 1.1, string.ascii_uppercase[lidx], transform=this_ax.transAxes, 
                size=20, weight='bold')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')   

# %%
if __name__ == "__main__":
    trmm=True
    anomalies=True

    grid_step = 1
    vname = 'pr'
    
    name='asia_es'   
    if trmm:
        grid_step = 0.5
        name='trmm_asia_es'

    num_cpus = 64
    num_jobs=16
    
    # Asia subcontinent range
    lat_range=[-15, 45]
    lon_range=[55, 150]
    if os.getenv("HOME") =='/home/goswami/fstrnad80':
        dirname = "/home/goswami/fstrnad80/data/GPCP/"
        if trmm is True:
            dirname = "/mnt/qb/goswami/fstrnad80/trmm/"

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


    job_id=1
    networkfile_lb = PATH + f"/../../outputs/{name}_ES_net_lb_{grid_step}.npz"
    networkfile_lb = PATH + f"/../../outputs/{name}_ES_net_{grid_step}.npz"

    sbm_filepath=PATH + f"/../graphs/{name}_{grid_step}/{job_id}_{name}_graph_tool_ES_{grid_step}"    
    
    print('Loading Data')
    load_ds = True
    dataset_file = PATH + f"/../../outputs/{name}_ds_{grid_step}.nc"
    if anomalies is True:
        dataset_file = PATH + f"/../../outputs/{name}_ds_{grid_step}_anomalies.nc"

    # %% 
    # Compare runs
    num_runs=10
    job_id=1
    sbm_filepath=PATH + f"/../graphs/{name}_{grid_step}/{job_id}_{name}_graph_tool_ES_{grid_step}"

    group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )    
    
    ds_cr = Analyse_Network(nc_file=dataset_file, name=name, 
                              network_file=networkfile_lb,
                              group_levels=group_levels,
                              load=True)
    
    
    # %%
    sel_m=['South India', 'South Asia']
    
    
    savepath=PATH + f"/../../plots/asia/time_series/{name}_density_plots_{sel_m}_regions.pdf"

    mi_indices=ds_cr.get_main_loc_gm(sel_m, den_th=.88, num_runs=30,
                                plot=True,savepath=savepath, projection='PlateCarree' )
    m_ts_dict=ds_cr.analyse_es_data( sel_m, mi_indices)

    # %%
    map1=m_ts_dict[sel_m[0]]['map']
    map2=m_ts_dict[sel_m[1]]['map']
    fig, ax= plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
    ds_cr.plot_contour_map(map1, color='black', cmap='Reds_r', projection='PlateCarree', 
                        fill_out=False, n_contour=2, vmin=0, vmax=1, bar=False, ax=ax)
    ds_cr.plot_contour_map(map2, color='black', cmap='Reds_r', projection='PlateCarree', 
                        fill_out=False, n_contour=2, vmin=0, vmax=1, bar=False, ax=ax)
    savepath=PATH + f"/../../plots/asia/time_series/{name}_{sel_m}_regions.pdf"
    plt.savefig(savepath)
    # %%
    import src.event_synchronization as es
    es_r1=m_ts_dict[sel_m[0]]['ts']
    es_r2=m_ts_dict[sel_m[1]]['ts']
    
    def count_ev_area(es_r, lp_filter=True):
        num_tp=es_r.shape[1]
        tp_es=es.prepare_es_input_data(es_r)
        ts_e=np.zeros(num_tp)
        for evs in tp_es:
            ts_e[evs]+=1
        
        if lp_filter is True:
            cutoff=4 # Attention this changes dramatically the smearing out of local extreme!
            fcutoff=.95 * 1. / cutoff
            order=8
            fs=1
            rp=.05    
            ts_e=es.cheby_lowpass_filter(ts_e, fcutoff, fs, order, rp)

        return ts_e

    m_times1=count_ev_area(es_r1, lp_filter=True)
    m_times2=count_ev_area(es_r2, lp_filter=True)
    
    
    def compute_lead_lag_corr(self, ts1, ts2, maxlags=20, corr_method='spearman', savepath=None, title=None):
        import scipy.stats as st
        Nx = len(ts1)
        if Nx != len(ts2):
            raise ValueError('x and y must be equal length')
        nts1, nts2=self.normalize_input(ts1,ts2)
        nts1, nts2=(ts1,ts2)
        
        corr_range=[]
        p_val_arr=[]

        if corr_method=='spearman':
            corr_func=st.stats.spearmanr
        elif corr_method=='pearson':
            corr_func=st.stats.pearsonr
        for lag in range(maxlags+1, -maxlags,-1):
            nts1_shift=np.roll(nts1, lag)
            corr, p_val=corr_func(nts1_shift, nts2)
            corr_range.append(corr)
            p_val_arr.append(p_val)
        
        
        fig, ax= plt.subplots(figsize=(8,5))
        x_lag=np.arange(-maxlags, maxlags+1) # because last value in arange is not counted
        ax.plot(x_lag, corr_range)
        ax1_2=ax.twinx()
        ax1_2.plot(x_lag, p_val_arr, color='red',ls='--')
        ax1_2.set_ylim(-0.,0.1)
        ax.grid()
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Correlation')
        ax1_2.set_ylabel('p-Value')
        ax1_2.yaxis.label.set_color('red')
        ax1_2.tick_params(axis='y', colors='red')
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath)
        return corr_range  

    method='pearson'
    savepath=PATH + f"/../../plots/asia/time_series/{method}_{name}_lead_lag_corr_{sel_m}.pdf"

    # corr=ds_cr.compute_lead_lag_corr( m_times1, m_times2, maxlags=20, title=sel_m, savepath=savepath)
    corr=compute_lead_lag_corr(ds_cr, m_times1, m_times2, maxlags=20, corr_method=method,
                              title=sel_m, savepath=savepath)


    # %%
    taumax=10
    q=0.99
    
    times=ds_cr.data_evs['time']

    savepath=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_synch_ts_sel_points_{sel_m}_q_{q}.pdf"
    ts12, ts21, tps1, tps2 = ds_cr.get_sync_times_monsoon(sel_m, m_ts_dict, taumax, q=q, plot=True, 
                                                lp_filter=False, rm=None, 
                                                savepath=savepath )
    
    savepath=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_{sel_m}_q_{q}.pdf"
    _=ds_cr.count_tps_occ([times[tps1], times[tps2]], label_arr=sel_m, savepath=savepath)                                                


    # %% 
    # Plot specific days
    grid_step=1
    plevel=250 
    dataset_file_an_gpcp = PATH + f"/../../outputs/gpcp_asia_es_ds_{grid_step}_anomalies.nc"
    ds_gpcp_an=BaseDataset(nc_file=dataset_file_an_gpcp, name=name, 
                  load=True, rrevs=False,)

    # Wind data
    w_name='wind_era5'
    if plevel==850:
        dataset_file_u = PATH + f"/../../outputs/{w_name}_u_ds_{grid_step}_anomalies.nc"
        dataset_file_v = PATH + f"/../../outputs/{w_name}_v_ds_{grid_step}_anomalies.nc"
    else:
        dataset_file_u = PATH + f"/../../outputs/{w_name}_u_{plevel}_ds_{grid_step}_anomalies.nc"
        dataset_file_v = PATH + f"/../../outputs/{w_name}_v_{plevel}_ds_{grid_step}_anomalies.nc"
    # ds_uwind = BaseDataset(dataset_file_u, load=load_ds, rrevs=False, name=w_name)
    # ds_vwind = BaseDataset(dataset_file_v, load=load_ds, rrevs=False, name=w_name)
    ds_wind=Wind_dataset(nc_file_u=dataset_file_u, nc_file_v=dataset_file_v, 
                            load=load_ds,name=w_name)
    # Surface pressure data
    psurf_name='psurf_era5'
    dataset_file_psurf = PATH + f"/../../outputs/{psurf_name}_sp_ds_{grid_step}_anomalies.nc"

    ds_psurf = BaseDataset(dataset_file_psurf, load=load_ds, rrevs=False, name=psurf_name)

    # %%

    

    map1=m_ts_dict[sel_m[0]]['map']
    map2=m_ts_dict[sel_m[1]]['map']


    tps=[tps1, tps2]
    tp_idx=1
    savepath=PATH + f"/../../plots/asia/time_series/{tp_idx+1}_{name}_pr_an_sync_times_{sel_m}_q_{q}.png"

    plot_day_progression(ds_cr, ds_cr.data_anomalies, times, tps[tp_idx], re_map1= map1, re_map2=map2, 
                        savepath=savepath, vmin=-7, vmax=7, label="Anomalies pr [mm/day]")

    # %% 
    # PSurf
    savepath=PATH + f"/../../plots/asia/time_series/{tp_idx+1}_{name}_psurf_an_sync_times_{sel_m}_q_{q}.png"

    plot_day_progression(ds_cr, ds_psurf.data_anomalies, times, tps[tp_idx], re_map1= map1, re_map2=map2, 
                        savepath=savepath, vmin=-100, vmax=100, label="Anomalies pSurf [Pa]")
    
    # %%
    # specific humidity
    name_sh='sh_era5'
    plevel=250
    dataset_file_sh = PATH + f"/../../outputs/{name_sh}_{plevel}_sp_ds_{grid_step}_anomalies.nc"

    ds_sh = BaseDataset(dataset_file_sh, load=load_ds, rrevs=False, name=name_sh)
    savepath=PATH + f"/../../plots/asia/time_series/{tp_idx+1}_{name}_sh_{plevel}_an_sync_times_{sel_m}_q_{q}.png"

    plot_day_progression(ds_cr, ds_sh.data_anomalies, times, tps[tp_idx], re_map1= map1, re_map2=map2, 
                        savepath=savepath, vmin=None, vmax=None, label=f"Anomalies specific humidity {plevel}")

    # %%
    # gp humidity
    name_gp='gp_era5'
    plevel=250
    dataset_file_gp = PATH + f"/../../outputs/{name_gp}_{plevel}_z_ds_{grid_step}_anomalies.nc"

    ds_gp = BaseDataset(dataset_file_gp, load=load_ds, rrevs=False, name=name_sh)
    savepath=PATH + f"/../../plots/asia/time_series/{tp_idx+1}_{name}_gp_{plevel}_an_sync_times_{sel_m}_q_{q}.png"

    plot_day_progression(ds_cr, ds_gp.data_anomalies, times, tps[tp_idx], re_map1= map1, re_map2=map2, 
                        savepath=savepath, vmin=None, vmax=None, label=f"Anomalies Geopotential {plevel}")
    # %% 
    # v wind
    savepath=PATH + f"/../../plots/asia/time_series/{tp_idx+1}_{name}_{plevel}_vwind_an_sync_times_{sel_m}_q_{q}.png"

    plot_day_progression(ds_cr, ds_wind.ds_vwind.data_anomalies, times, tps[tp_idx], re_map1= map1, re_map2=map2, 
                        savepath=savepath, vmin=-1.5, vmax=1.5, label=f"Anomalies {plevel} v wind speed [m/s]")
    # %%
    # u wind
    savepath=PATH + f"/../../plots/asia/time_series/{tp_idx+1}_{name}_{plevel}_uwind_an_sync_times_{sel_m}_q_{q}.png"

    plot_day_progression(ds_cr, ds_wind.ds_uwind.data_anomalies, times, tps[tp_idx], re_map1= map1, re_map2=map2, 
                        savepath=savepath, vmin=-1.5, vmax=1.5, label=f"Anomalies {plevel} u wind speed [m/s]")
    # %% 
    # wind speed
    savepath=PATH + f"/../../plots/asia/time_series/{tp_idx+1}_{name}_{plevel}_wind_speed_an_sync_times_{sel_m}_q_{q}.png"

    plot_day_progression(ds_cr, ds_wind.windspeed_anomalies, times, tps[tp_idx], re_map1= map1, re_map2=map2, 
                        savepath=savepath, vmin=-1.5, vmax=1.5, label=f"Anomalies {plevel} wind speed [m/s]")



    # %% 
    # Times per Year
    savepath=PATH + f"/../../plots/asia/time_series/occ_pr_sync_times_year_{sel_m}_q_{q}.png"
    ds_cr.count_tps_per_year([times[tps1],times[tps2] ], label_arr=sel_m, savepath=savepath)
    # %%

    # %% 
    # Times per Day
    
    _=ds_cr.count_tps_occ_day(ds_cr, times[tps1], label_arr=sel_m, savepath=None)
