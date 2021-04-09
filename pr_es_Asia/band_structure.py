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
import random

from src.es_graph_tool import ES_Graph_tool
from src.compare_runs import Compare_Runs
from src.analyse_network import Analyse_Network
from src.monsoon_region_es import Monsoon_Region_ES
from src.dataset import BaseDataset
from src.pr_es_Asia.wind_presure_dataset import Wind_dataset
import src.pr_es_Asia.tp_analysis_asia as tpa

colors=['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'm', 'c', 'y']

# %%
if __name__ == "__main__":
    trmm=True
    anomalies=True

    grid_step = 0.5
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
    job_id=1
    
    taumax=10
    q=0.99

    clat_arr=np.arange (15,20, 5)
    # %%
    sbm_filepath=PATH + f"/../graphs/{name}_{grid_step}/{job_id}_{name}_graph_tool_ES_{grid_step}"

    group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )    
    
    ds_cr = Analyse_Network(nc_file=dataset_file, name=name, 
                              network_file=networkfile_lb,
                              group_levels=group_levels,
                              load=True)
    times=ds_cr.data_evs['time']
    # %%
    savepath=PATH + f"/../../plots/asia/entropy_num_groups_multiple_runs_asia.pdf"
    import src.link_bundles as lb
    def compare_entropy(self, num_runs=10, max_num_levels=14, plot=False, savepath=None, graph_file=None, ax=None):
        
        sbm_entropy_arr=np.zeros((num_runs, max_num_levels))
        sbm_num_groups_arr=np.zeros((num_runs, max_num_levels))
        
        for idx, job_id in enumerate(range(0,num_runs)):
            if graph_file is None:
                sbm_filepath=self.PATH + f"/graphs/{self.dataset_name}_{self.grid_step}/{job_id}_{self.dataset_name}_graph_tool_ES_{self.grid_step}"
            else:
                sbm_filepath=self.PATH + graph_file

            if not os.path.exists(sbm_filepath +'_group_levels.npy'):
                print(f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
                continue
            sbm_entropy=np.load(sbm_filepath+'_entropy.npy',  allow_pickle=True )
            sbm_num_groups=np.load(sbm_filepath+'_num_groups.npy',  allow_pickle=True )

            sbm_entropy_arr[idx,:len(sbm_entropy)]=sbm_entropy
            sbm_num_groups_arr[idx,:len(sbm_num_groups)]=sbm_num_groups

            if plot is True:
                self.plot_entropy_groups(entropy_arr=sbm_entropy, groups_arr=sbm_num_groups)

        mean_entropy, std_entropy, _, _, _, _, _ = lb.compute_stats(runs=sbm_entropy_arr)
        mean_num_groups, std_num_groups, _, _, _, _, _ = lb.compute_stats(runs=sbm_num_groups_arr)
        # -1 Because last level is trivial!
        mean_entropy= mean_entropy[:-1]
        std_entropy= std_entropy[:-1]
        mean_num_groups= mean_num_groups[:-1]
        std_num_groups= std_num_groups[:-1]

        # Now plot
        from matplotlib.ticker import MaxNLocator
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
        num_levels=len(mean_entropy)
        ax.set_xlabel('Level')

        # Entropy
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(rf'Description Length $\Gamma$')
        
        x_data=np.arange(1,num_levels+1)
        ax.errorbar(x_data, (mean_entropy), yerr=(std_entropy), 
                            color='tab:blue', elinewidth=2,label='Descr. Length')
        ax.fill_between(x_data, mean_entropy - std_entropy, mean_entropy + std_entropy,
                        color='tab:blue', alpha=0.3)
        ax.set_yscale('log')
        ax.yaxis.label.set_color('tab:blue')
        ax.tick_params(axis='y', colors='tab:blue')
    
        # Number of Groups
        ax1_2=ax.twinx()
        ax1_2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1_2.set_ylabel('Number of groups')
        ax1_2.errorbar(x_data, mean_num_groups, yerr=std_num_groups,
                                color='tab:green', label='Groups')
        ax1_2.fill_between(x_data, mean_num_groups - std_num_groups, mean_num_groups + std_num_groups,
                        color='tab:green', alpha=0.3)
        ax1_2.set_yscale('log')
        ax1_2.yaxis.label.set_color('tab:green')
        ax1_2.tick_params(axis='y', colors='tab:green')
    
        ax.legend(loc='upper right', bbox_to_anchor=(.4,1), bbox_transform=ax.transAxes)
        ax1_2.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        
        # fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)



        if savepath is not None:
            print(f"Store files to: {savepath}")
            fig.savefig(savepath, bbox_inches='tight')

        return mean_entropy, std_entropy, mean_num_groups, std_num_groups

    # mean_entr, std_entr, mean_ngroups, std_ngroups= compare_entropy(ds_cr, num_runs=30,max_num_levels=8,
    #                                                                       savepath=savepath)
    
    # %%
    def compute_bands_idx_dict(ds, clat_arr, lon_range=None, bands='cluster', colors=colors):
        cluster_idx_dict=dict()

        for idx, clat in enumerate(clat_arr):
            loc_arr=[]
            for clon in range(80, 150, 20):
                loc_arr.append([clat, clon])

            if bands=='cluster':
                savepath_region=PATH + f"/../../plots/asia/time_series/{name}_density_plots_{clat}_regions.pdf"
                c_indices=ds.get_density_map_loc_arr(loc_arr, den_th=.88, num_runs=30, 
                                                plot=True,savepath=savepath_region, projection='PlateCarree' )
            elif bands=='lat_bands':
                offset=2.5
                if idx==0:
                    lat_range=np.array([clat-5, clat])
                else:
                    lat_range=np.array([clat_arr[idx-1], clat])
                if lon_range is None:
                    lon_range = ds.lon_range
                
                lat_range=lat_range+offset
                mask=ds.mask
                c_indices=ds.get_idx_range(def_map=mask,lon_range=lon_range, lat_range=lat_range)
            else:
                raise ValueError(f"This method {bands} is not available!")

            cluster_idx_dict[clat]={'indices': c_indices, 'color':colors[idx]}

        return cluster_idx_dict

    cluster_idx_dict=compute_bands_idx_dict(ds=ds_cr, clat_arr=clat_arr, bands='cluster')

     # %% 
    # run ensemble mean of samples from cluster
    num_samples=200
    num_runs=100

     

    def sample_from_cluster(ds, taumax, q, num_samples, num_runs, times, ts_indices):
        tps_arr=[]
        tps_idx=[]
        for run in range(num_runs):
            sample_ts_indices=ts_indices[random.sample(list(np.arange(len(ts_indices))),k=num_samples)]
            savepath=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_synch_ts_sel_points_region_q_{q}.pdf"
            ts12, ts21, tps1, tps2 = ds.get_sync_times(es_r1=sample_ts_indices, es_r2=sample_ts_indices, 
                                                        taumax=taumax, q=q, plot=False, 
                                                        lp_filter=False, rm=None, 
                                                        savepath=savepath )
            tps_arr.append(times[tps1])
            tps_idx.append(tps1)
        return tps_arr, tps_idx

    def compute_es_ensemble_bands(ds, clat_arr, num_samples=num_samples, num_runs=num_runs, bands='cluster'):

        mean_cnt_arr=[]
        std_cnt_arr=[]
        
        savepath_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_ensemble_{bands}.npy"

        cluster_idx_dict=compute_bands_idx_dict(ds, clat_arr=clat_arr, bands=bands)
        for idx, clat in enumerate(clat_arr):

            c_indices=cluster_idx_dict[clat]['indices']
            savepath_tps_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_ensemble_{bands}_tps_{clat}.npy"

            ts_indices=ds.get_ts_of_indices(c_indices)
            tps_arr,tps1= sample_from_cluster(ds, taumax, q, num_samples, num_runs, times, ts_indices)
            savepath_cnt=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_ensemble_q_{q}_{clat}_{bands}.pdf"
            all_run_cnt=ds.count_tps_occ(tps_arr, plot=False, label_arr=None, savepath=savepath_cnt)   

            mean_cnt=np.mean(all_run_cnt,axis=0)
            std_cnt=np.std(all_run_cnt,axis=0)
            mean_cnt_arr.append(mean_cnt)
            std_cnt_arr.append(std_cnt)
            
            np.save(savepath_np, np.array([mean_cnt_arr, std_cnt_arr], dtype=object))
            np.save(savepath_tps_np, np.array(tps1, dtype=object))
        return mean_cnt_arr, std_cnt_arr

    compute_es_ensemble_bands(ds=ds_cr, clat_arr=clat_arr, num_samples=num_samples, num_runs=num_runs, bands='cluster')
    sys.exit(0)
    # %% 
    # Single lat

    def get_sync_times(ds, c_indices, clat=None,
                       taumax=10, q=0.99, lp_filter=False, rm=None, plot=True, savepath = None):
        """
        Get sync times for two regions within sel_m
        """
        times=ds.data_evs['time']
        ts_indices=ds_cr.get_ts_of_indices(c_indices[:])

        sync_times12, sync_times21, ts12, ts21=ds.get_sync_times_2mregions(es_r1=ts_indices, es_r2=ts_indices, 
                                                                           taumax=taumax, q=q, 
                                                                           lp_filter=lp_filter, rm=rm)
        
        tps1=times[sync_times12]
        tps2=times[sync_times21]

        if plot is True:
            fig,ax=plt.subplots(figsize=(7,5), nrows=1, ncols=1)
            ax.plot(times, ts12, label='Synchronous Events')
            ax.plot(tps1, ts12[sync_times12],'x', label="Selected time points")
            
            ax.set_title(f"Band {clat}°N")
            ax.set_ylabel(f'# Associated Sync Events')

            fig.legend(bbox_to_anchor=(.8, .9), loc='upper left', 
            fancybox=True, shadow=True, ncol=1)
            fig.tight_layout()
            if savepath is not None:
                fig.savefig(savepath, bbox_inches='tight')
            
        return ts12, ts21, sync_times12, sync_times21,
    

    clat=15
    savepath=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_q_{q}_time_series_{clat}.pdf"

    c_indices=cluster_idx_dict[clat]['indices']
    _=get_sync_times(ds_cr, c_indices, clat=clat, savepath=savepath) 

    # %%
    # All points
    def compute_es_allpoints_bands(ds, clat_arr, bands='cluster'):
        # Take all points in the cluster and compute the specific dates of synchronization
        times=ds.data_evs['time']

        cluster_idx_dict=compute_bands_idx_dict(ds, clat_arr=clat_arr, bands=bands)
        mean_cnt_arr=[]
        std_cnt_arr=[]
        tps_arr=[]
        savepath_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_all_points.npy"
        savepath_tps_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_all_points_tps.npy"

        for idx, clat in enumerate(clat_arr):
            
            c_indices=cluster_idx_dict[clat]['indices']

            num_indices = 400
            if len(c_indices)<num_indices:
                num_indices=len(c_indices)
            ts_indices=ds_cr.get_ts_of_indices(c_indices)
            
            ts12, ts21, tps1, tps2 = ds_cr.get_sync_times(es_r1=ts_indices, es_r2=ts_indices, 
                                                    taumax=taumax, q=q, plot=False, 
                                                    lp_filter=False, rm=None, 
                                                    savepath=None )
            savepath=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_allpoints_q_{q}_{clat}_{bands}.pdf"
            count_occ=ds_cr.count_tps_occ([times[tps1]], label_arr=None, savepath=savepath)
            
            mean_cnt=np.mean(count_occ,axis=0)
            std_cnt=np.std(count_occ,axis=0)
            mean_cnt_arr.append(mean_cnt)
            std_cnt_arr.append(std_cnt)
            tps_arr.append(tps1)

        np.save(savepath_np, np.array([mean_cnt_arr, std_cnt_arr], dtype=object))
        np.save(savepath_tps_np, np.array(tps_arr, dtype=object))

        return mean_cnt_arr, std_cnt_arr, tps_arr
                    
    mean_cnt_arr, std_cnt_arr, tps_arr =compute_es_allpoints_bands(ds=ds_cr, clat_arr=clat_arr, bands='cluster')


    # %% 
    # Plot results
    
    def plot_cnt_occ_ensemble(ds, mean_cnt_arr,std_cnt_arr, savepath, 
                            colors=['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'm', 'c', 'tab:brown'],
                            polar=True ):
        if polar is True:
            fig, ax=plt.subplots(figsize=(9,6),subplot_kw={'projection': 'polar'})
            ax.margins(y=0)
            x_pos=np.deg2rad(np.linspace(0,360,13))
            ax.set_xticks(x_pos )
            ax.set_xticklabels(ds.months+[''], )
            ax.set_rlabel_position(60)  # get radial labels away from plotted line
            ax.set_rticks([0., 0.1, 0.2, .3, 0.4])  # Less radial ticks
            ax.set_theta_offset(np.pi) # rotate the axis arbitrarily, just replace pi with the angle you want.
        else:
            fig, ax=plt.subplots(figsize=(8,5))
            ax.set_xlabel('Month')
            ax.set_ylabel('Relative Frequency')
            x_pos=ds.months
        
        if len(mean_cnt_arr) != len(std_cnt_arr):
            raise ValueError(f"Mean len {len(mean_cnt_arr)} != Std len {len(std_cnt_arr)}")
        for idx in range(len(mean_cnt_arr)):
            print(idx)
            mean_cnt=np.array(mean_cnt_arr[idx],dtype=float)
            std_cnt=np.array(std_cnt_arr[idx],dtype=float)
            if polar is True:
                mean_cnt=np.append(mean_cnt, np.array([mean_cnt[0]]), axis=0)
                std_cnt=np.append(std_cnt, np.array([std_cnt[0]]), axis=0)

            ax.errorbar(x_pos, mean_cnt , yerr=(std_cnt), 
                        color=colors[idx],elinewidth=2,label=f"Band {clat_arr[idx]}°N")
            ax.fill_between(x_pos, mean_cnt - std_cnt, mean_cnt + std_cnt,
                        color=colors[idx], alpha=0.3)

        
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left', 
        fancybox=True, shadow=True, ncol=1)

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight')
    
    bands='cluster'
    savepath_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_ensemble_bak.npy"
    savepath_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_ensemble_{bands}.npy"
    # savepath_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_{bands}.npy"

    load_mean, load_std=np.load(savepath_np,allow_pickle=True)

    savepath=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_ensemble_{bands}.pdf"
    load_mean, load_std=np.load(savepath_np,allow_pickle=True)
    plot_cnt_occ_ensemble(ds_cr, load_mean, load_std, savepath=savepath, polar=False, colors=colors)
    
    

   

    

    # %%
    
    def plt_mask_on_map(self, ax, projection):
        left_out=xr.where(np.isnan(self.mask), 1, np.nan)
        ax.contourf(self.dataarray.coords['lon'], self.dataarray.coords['lat'], 
                    left_out,2, hatches=[ '...', '...',], colors='none',extend='lower',
                    transform=projection)

    def plot_contour_map(self, dmap, n_contour=1, central_longitude=0, vmin=None, vmax=None,
                 fig=None, ax=None, projection='Mollweide', cmap=None, color=None, bar=True, plt_mask=False,
                 ticks=None, clabel=False, fill_out=True, points=False, label=None, extend='both', title=None):
        """Simple map plotting using xArray.
        
        Args:
        -----
        dmap: xarray
        
        """
        import matplotlib.ticker as mticker
        import warnings
        warnings.filterwarnings("ignore")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(9,6))
            if projection =='Mollweide':
                proj=ccrs.Mollweide(central_longitude=central_longitude)
            elif projection=='PlateCarree':
                proj=ccrs.PlateCarree(central_longitude=central_longitude)
            else:
                raise ValueError(f'This projection {projection} is not available yet!')

            ax = plt.axes(projection=proj)
        ax.coastlines()
        ax.add_feature(ctp.feature.BORDERS, linestyle=':')
        gl=ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, )
        
        gl.left_labels=True
        gl.right_labels=False
        if projection=='PlateCarree':
            gl.top_labels = True
            gl.bottom_labels = False
        
        projection = ccrs.PlateCarree(central_longitude=central_longitude)
        if cmap is not None:
            cmap=plt.get_cmap(cmap)
        elif color is None:
            raise ValueError("Please select either color or cmap! None is chosen yet!")

        ax.set_extent([np.min(dmap.coords['lon']), 
                      np.max(dmap.coords['lon']),
                      np.min(dmap.coords['lat']),
                      np.max(dmap.coords['lat'])], crs=projection) 
    
        X, Y = np.meshgrid(dmap.coords['lon'], dmap.coords['lat'])
        data_0=xr.where(np.isnan(dmap.data), 0, dmap.data)   

        levels=None
        if vmin is not None and vmax is not None:
            levels = np.linspace(vmin, vmax, n_contour)
            data_0=xr.where(data_0>vmax, vmax, data_0)
            data_0=xr.where(data_0<vmin, vmin, data_0)
            
        else:
            levels=n_contour
        
        if fill_out is True:
            cbar = ax.contourf(X, Y, data_0, levels=levels,vmin=vmin, vmax=vmax,
                cmap=cmap, transform=projection,
                )
            
        else:
            cbar = ax.contour(X, Y, data_0, levels=levels,vmin=vmin, vmax=vmax,
                    cmap=cmap, transform=projection, linewidths=3, colors=color,
                    )
            if points is True:
                flat_idx_lst=self.flatten_array(dmap, time=False, check=False)
                flat_idx=np.where(flat_idx_lst>0)[0]
                loc_list=[]
                xp=[]
                yp=[]
                for idx in flat_idx:
                    map_idx=self.get_map_index(idx)
                    xp.append(map_idx['lon'])
                    yp.append(map_idx['lat'])
                
                ax.plot(xp, yp,
                    color=color, linewidth=0, marker='.', transform=projection,alpha=0.2
                    )

        if plt_mask:
            plt_mask_on_map(self, ax=ax,projection=projection)       
            
        if clabel is True:
            ax.clabel(cbar, inline=1, fmt='%1d', fontsize=14, colors='k'  )

        if bar:
            if label is None:
                label=dmap.name
            if fig is not None:
                cbar=self.make_colorbar(ax, cbar, orientation='horizontal', label=label,ticks=ticks,
                                extend=extend)

        if title is not None:
            y_title=1.1
            ax.set_title(title)

        return ax, cbar



    bands='cluster'
    # cluster_idx_dict=compute_bands_idx_dict(ds=ds_cr, clat_arr=clat_arr, bands=bands)

    from matplotlib.patches import Rectangle
    fig, ax= plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
    legend_items=[]
    legend_item_names=[]
    for clat, clat_dict in cluster_idx_dict.items():
        c_indices=clat_dict['indices']
        map=ds_cr.get_map(ds_cr.flat_idx_array(c_indices))
        color=clat_dict['color']
        
        ax,_=plot_contour_map(ds_cr, map, color=color,
                            projection='PlateCarree', points=True,plt_mask=True,
                            fill_out=False, n_contour=2, vmin=0, vmax=2, bar=False,
                            ax=ax)

        legend_items.append(Rectangle((0, 0), 1, 1, fc=color, alpha=0.2, fill=True, 
                            edgecolor=color, linewidth=2) )
        legend_item_names.append(f"Band {clat} °N")
    ax.legend( legend_items, legend_item_names ,
            bbox_to_anchor=(1, 1), loc='upper left', 
            fancybox=True, shadow=True, ncol=1)
    savepath=PATH + f"/../../plots/asia/time_series/{name}_{bands}.pdf"
    plt.savefig(savepath, bbox_inches='tight')
    

    # %% 
    # For all points
    grid_step=.5

    path_tps_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_all_points_tps.npy"
    tps_arr = np.load(path_tps_np, allow_pickle=True)
    def update_cluster_idx_dict(cluster_idx_dict=None):
        if cluster_idx_dict is None:
            cluster_idx_dict=dict()
        for idx, clat in enumerate(clat_arr):    
            cluster_idx_dict[clat]=tps_arr[idx]
        return cluster_idx_dict
    
    cluster_idx_dict= update_cluster_idx_dict()
    # %% 
    # For ensemble analysis
    grid_step=.5
    clat=15
    path_tps_np=PATH + f"/../../plots/asia/time_series/{name}_{grid_step}_count_sync_times_q_{q}_ensemble_cluster_tps_{clat}.npy"
    tps_arr = np.array(np.load(path_tps_np, allow_pickle=True))
    tps=[]
    for tp in tps_arr:
        tps.append(tp)        
    
    # %%
    # Plot days 
    plevel=850
    clat=15
    tps = cluster_idx_dict[clat]
    grid_step=1
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
    
    # %%
    # Rainfall data   

    def plot_day_progression(ds, da, ds_wind, times, tps, re_map1=None, re_map2=None,  vmin=-5, vmax=5, label=None, savepath=None):
        ncols=2
        nrows= 4
        projection=ccrs.PlateCarree()
        import string
        fig,ax=plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols,5*nrows),
                            subplot_kw=dict(projection=projection ) )
        
        day_offset=-2
        for lidx, day in enumerate(range(day_offset,ncols*nrows + day_offset)):
            i = int(lidx/ncols)
            j= lidx-ncols*i
            this_ax=ax[i][j]

            title=f"Day {day}"
            bar_plot=False
            if day>= 4:
                bar_plot=True
            an_map=da.sel(time=times[tps+day], method='nearest').mean(dim='time')
            # print("Maximum:", float(an_map.max()) )
            ds.plot_map(an_map, ax=this_ax, title=title,label=label, plt_mask=False,bar=bar_plot,
                        projection='PlateCarree', color='coolwarm_r', vmin=vmin, vmax=vmax)

            u=ds_wind.ds_uwind.data_anomalies.sel(time=times[tps+day], method='nearest').mean(dim='time') 
            v=ds_wind.ds_vwind.data_anomalies.sel(time=times[tps+day], method='nearest').mean(dim='time')
            this_ax=tpa.plot_wind_field_ax(ax=this_ax, u=u, v=v, )

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
     
    savepath=PATH + f"/../../plots/asia/time_series/{name}_pr_an_sync_times_q_{q}_lat_{clat}.png"

    plot_day_progression(ds_cr, ds_cr.data_anomalies, ds_wind=ds_wind, times=times,tps=tps,  
                        savepath=savepath, vmin=-7, vmax=7, label="Anomalies pr [mm/day]")


    # %% 
    # Surface pressure data
    psurf_name='psurf_era5'

    dataset_file_psurf = PATH + f"/../../outputs/{psurf_name}_sp_ds_{grid_step}_anomalies.nc"

    ds_psurf = BaseDataset(dataset_file_psurf, load=load_ds, rrevs=False, name=psurf_name)
    
    savepath=PATH + f"/../../plots/asia/time_series/{name}_psurf_an_sync_times_q_{q}_lat_{clat}.png"

    plot_day_progression(ds_cr, ds_psurf.data_anomalies, ds_wind=ds_wind, times=times, tps=tps,  
                        savepath=savepath, vmin=-300, vmax=300, label="Anomalies surface pressure [Pa]")
    
    # %%
    # specific humidity
    name_sh='sh_era5'
    dataset_file_sh = PATH + f"/../../outputs/{name_sh}_{plevel}_sp_ds_{grid_step}_anomalies.nc"

    ds_sh = BaseDataset(dataset_file_sh, load=load_ds, rrevs=False, name=name_sh)
    savepath=PATH + f"/../../plots/asia/time_series/{name}_sh_{plevel}_an_sync_times_q_{q}_lat_{clat}.png"

    plot_day_progression(ds_cr, ds_sh.data_anomalies, ds_wind=ds_wind, times=times, tps=tps, 
                        savepath=savepath, vmin=None, vmax=None, label=f"Anomalies specific humidity {plevel}")

    # %%
    # gp 
    name_gp='gp_era5'
    dataset_file_gp = PATH + f"/../../outputs/{name_gp}_{plevel}_z_ds_{grid_step}_anomalies.nc"

    ds_gp = BaseDataset(dataset_file_gp, load=load_ds, rrevs=False, name=name_gp)
    savepath=PATH + f"/../../plots/asia/time_series/{name}_gp_{plevel}_an_sync_times_q_{q}_{clat}.png"

    plot_day_progression(ds_cr, ds_gp.data_anomalies, ds_wind=ds_wind, times=times, tps=tps, 
                        savepath=savepath, vmin=-300, vmax=300, label=f"Anomalies Geopotential {plevel} hPa")
    # %% 
    # v wind
    savepath=PATH + f"/../../plots/asia/time_series/{name}_{plevel}_vwind_an_sync_times_q_{q}_{clat}.png"

    plot_day_progression(ds_cr, ds_wind.ds_vwind.data_anomalies, ds_wind=ds_wind, times=times, tps=tps, 
                        savepath=savepath, vmin=-1.5, vmax=1.5, label=f"Anomalies {plevel} v wind speed [m/s]")
    # %%
    # u wind
    savepath=PATH + f"/../../plots/asia/time_series/{name}_{plevel}_uwind_an_sync_times_q_{q}_{clat}.png"

    plot_day_progression(ds_cr, ds_wind.ds_uwind.data_anomalies, ds_wind=ds_wind, times=times, tps=tps, 
                        savepath=savepath, vmin=-1.5, vmax=1.5, label=f"Anomalies u wind speed [m/s] at {plevel} hPa")
    # %% 
    # wind speed
    savepath=PATH + f"/../../plots/asia/time_series/{name}_{plevel}_wind_speed_an_sync_times_q_{q}_{clat}.png"

    plot_day_progression(ds_cr, ds_wind.windspeed_anomalies, ds_wind=ds_wind, times=times, tps=tps, 
                        savepath=savepath, vmin=-3.5, vmax=3.5, label=f"Anomalies wind speed [m/s] at {plevel} hPa")

