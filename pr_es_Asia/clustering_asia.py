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
    run_gt=False

    grid_step = 1
    vname = 'pr'
    anomalies=True
    
    name='asia_es'   
    if trmm:
        grid_step =0.5
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
        if grid_step <0.5:
            networkfile_lb = PATH + f"/../../outputs/{name}_ES_net_{grid_step}.npz"


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
        job_id=1

    networkfile_lb = PATH + f"/../../outputs/{name}_ES_net_lb_{grid_step}.npz"

    sbm_filepath=PATH + f"/../graphs/{name}_{grid_step}/{job_id}_{name}_graph_tool_ES_{grid_step}"    
    
    print('Loading Data')
    load_ds = True
    dataset_file = PATH + f"/../../outputs/{name}_ds_{grid_step}.nc"
    if anomalies is True:
        dataset_file = PATH + f"/../../outputs/{name}_ds_{grid_step}_anomalies.nc"
    

    # %%
    ds_gt = ES_Graph_tool( nc_file=dataset_file, name=name, network_file=networkfile_lb,
                            load=True,rrevs=False)

    # %%
    if run_gt:
        state, group_levels, sbm_matrix_arr, entropy_arr,num_groups_arr, h= ds_gt.apply_SBM(g=ds_gt.graph, 
                                                                        B_max=50,
                                                                        savepath=sbm_filepath, parallel=True,
                                                                        epsilon=1e-1)
        sys.exit(0)

    # %%
    # Visualize single cluster run
    
    sbm_filepath=PATH + f"/../graphs/{name}_{grid_step}/{job_id}_{name}_graph_tool_ES_{grid_step}"
    if not os.path.exists(sbm_filepath +'_group_levels.npy'):
        raise ValueError(f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
    sbm_matrix_arr=np.load(sbm_filepath+'_sbm_matrix.npy',  allow_pickle=True )
    sbm_group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )
    sbm_entropy=np.load(sbm_filepath+'_entropy.npy',  allow_pickle=True )
    sbm_num_groups=np.load(sbm_filepath+'_num_groups.npy',  allow_pickle=True )
    ds_gt.plot_entropy_groups(entropy_arr=sbm_entropy, groups_arr=sbm_num_groups)

    
    node_levels, sbm_node_counts=ds_gt.node_level_arr(sbm_group_levels)
    
    ordered_node_levels, nl_lat_dict=ds_gt.parallel_ordered_nl_loc(node_levels)
    # %%
    lidx=2
    level=ordered_node_levels[lidx]
    map=ds_gt.get_map(level)
    
    def discrete_cmap(self,vmin, vmax, colormap=None, num_ticks=None, shift_ticks=False):
        import palettable.colorbrewer.diverging as pt
        import palettable.colorbrewer.qualitative as pt
        import matplotlib as mpl

        # colormap=pt.Spectral_11.mpl_colormap
        if colormap is None:
            colormap=pt.Paired_12.mpl_colormap
        cmap=plt.get_cmap(colormap)

        normticks=self.discrete_norm_ticks(vmin, vmax, num_ticks=num_ticks, shift_ticks=shift_ticks)

        norm = mpl.colors.BoundaryNorm(normticks, cmap.N)
        return cmap, norm
    
    def discrete_norm_ticks(self,vmin, vmax, shift_ticks=False, num_ticks=None):
        if vmin is None or vmax is None:
            return None
        if num_ticks is None:
            num_ticks=vmax
        step=1
        max_ticks=15
        if shift_ticks is True:
            normticks=np.arange(vmin,vmax+1,step)
        else:
            normticks=np.arange(vmin,vmax+2,step)
        if num_ticks>=max_ticks:
            normticks=np.linspace(vmin,vmax+1,max_ticks,dtype=int)
        else:
            normticks=np.linspace(vmin,vmax,num_ticks+1,dtype=int)

        if vmax==2:
            normticks=np.array([0, 1, 2])
        elif (vmax) < 2:
            normticks=np.array([0, 1])
            print(f"WARNING! Number of clusters: {vmax} < 2!")
        
        return normticks


    def plot_map(self, dmap=None, central_longitude=0, vmin=None, vmax=None,
                 fig=None, ax=None, projection='PlateCarree', color='coolwarm_r', 
                 dcbar=False , bar=True,  num_ticks=None, shift_ticks=False, label=None, plt_mask=True,
                 extend='both', title=None):
        """Simple map plotting using xArray.
        
        Args:
        -----
        dmap: xarray
        
        """
        import matplotlib.ticker as mticker
        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 20
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        if ax is None:
#            ax = plt.subplot(projection=ccrs.Mollweide(central_longitude=central_longitude))
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


        if dmap is None:
            ax.set_extent([np.min(self.dataarray.coords['lon']), 
                      np.max(self.dataarray.coords['lon']),
                      np.min(self.dataarray.coords['lat']),
                      np.max(self.dataarray.coords['lat'])], crs=projection) 
            return ax

        ax.set_extent([np.min(dmap.coords['lon']), 
                      np.max(dmap.coords['lon']),
                      np.min(dmap.coords['lat']),
                      np.max(dmap.coords['lat'])], crs=projection) 

        normticks=[]
        ticks=None
        if dcbar is True:
            if vmin is None:
                vmin=int(np.nanmin(dmap.data))
            if vmax is None:
                vmax=int(np.nanmax(dmap.data))
            vmax +=1
            normticks= discrete_norm_ticks(self, vmin, vmax,  num_ticks=num_ticks, shift_ticks=shift_ticks)
            cmap, norm= discrete_cmap(self, vmin,vmax, colormap=color, num_ticks=num_ticks, shift_ticks=shift_ticks)
            if shift_ticks is True:
                normticks=normticks[:-1]
                ticks=normticks[:] +0.5
            else:
                ticks=normticks[:]

            cbar = ax.pcolormesh(
                    dmap.coords['lon'], dmap.coords['lat'], dmap.data,
                    cmap=cmap, vmin=vmin, vmax=vmax, transform=projection, norm=norm,
                    )
      
        else:
            cmap = plt.get_cmap(color)
            cbar = ax.pcolormesh(
                    dmap.coords['lon'], dmap.coords['lat'], dmap.data,
                    cmap=cmap, vmin=vmin, vmax=vmax, transform=projection, 
                    )

        if plt_mask:
            self.plt_mask_on_map(ax,projection=projection)

        if bar:
            if label is None:
                label=dmap.name
            cbar=self.make_colorbar(ax, cbar, orientation='horizontal', label=label,ticks=ticks,
                                extend=extend)
            if dcbar is True:
                string_ticks=[f"{x}" for x in normticks[:]]
                cbar.ax.set_xticklabels(string_ticks)

         
                             
        if title is not None:
            y_title=1.1
            ax.set_title(title)

        return ax

    savepath=PATH + f"/../../plots/asia/cluster_analysis/{job_id}_level_{lidx}_groups.png"
    plot_map(ds_gt, map, dcbar=True, title=f"Level {lidx+1}, #JobID{job_id}", projection='PlateCarree',
             extend='neither', color=None, shift_ticks=True, bar=True)
    # plt.savefig(savepath)

    # %%
    # All levels
    import cartopy.crs as ccrs
    import cartopy as ctp

    import src.link_bundles as lb
    def compare_entropy(self, num_runs=10, max_num_levels=14, plot=False, savepath=None, graph_file=None, ax=None):
        
        sbm_entropy_arr=np.zeros((num_runs, max_num_levels))
        sbm_num_groups_arr=np.zeros((num_runs, max_num_levels))
        
        for idx, job_id in enumerate(range(0,num_runs)):
            if graph_file is None:
                sbm_filepath=PATH + f"/graphs/{self.dataset_name}_{self.grid_step}/{job_id}_{self.dataset_name}_graph_tool_ES_{self.grid_step}"
            else:
                sbm_filepath=PATH + graph_file

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
                            color='tab:blue', elinewidth=2,label='Entropy')
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
    
        ax.legend(loc='upper right', bbox_to_anchor=(.25,1), bbox_transform=ax.transAxes)
        ax1_2.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        
        # fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)



        if savepath is not None:
            print(f"Store files to: {savepath}")
            fig.savefig(savepath)

        return mean_entropy, std_entropy, mean_num_groups, std_num_groups

    
    mean_entr, std_entr, mean_ngroups, std_ngroups= compare_entropy(ds_gt, num_runs=30,max_num_levels=8,
                                                                          savepath=None, ax=this_ax)

    # %%
    show_levels=7

    projection=ccrs.PlateCarree()
    ncols=2
    nrows= 4
    import string
    fig,ax=plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,6*nrows),
                           subplot_kw=dict(projection=projection ) )
    for lidx, level in enumerate(ordered_node_levels[-1-show_levels:-1]):
        i = int(lidx/ncols)
        j= lidx-ncols*i
        this_ax=ax[i][j]

        map=ds_gt.get_map(level)
        title=f"Level {lidx +1}"
        plot_map(ds_gt, map, ax=this_ax, dcbar=True, title=title, plt_mask=True,shift_ticks=True, color=None,
                        projection='PlateCarree', fig=fig, extend='neither')
        
        this_ax.text(-0.1, 1.1, string.ascii_uppercase[lidx], transform=this_ax.transAxes, 
                size=20, weight='bold')


    savepath=PATH + f"/../../plots/asia/cluster_analysis/{job_id}_{name}_{grid_step}_last_{show_levels}_levels.pdf"
    plt.savefig(savepath, bbox_inches = 'tight')

    # %% 
    # Visualize Clusters all runs
    
    for idx, job_id in enumerate(range(0,30)):
        print("Start Reading data jobid {job_id}...")
        sbm_filepath=PATH + f"/../graphs/{name}/{job_id}_{name}_graph_tool_ES_{grid_step}"
        if not os.path.exists(sbm_filepath +'_group_levels.npy'):
            print(f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
            continue
        sbm_matrix_arr=np.load(sbm_filepath+'_sbm_matrix.npy',  allow_pickle=True )
        sbm_entropy=np.load(sbm_filepath+'_entropy.npy',  allow_pickle=True )
        sbm_group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )
        sbm_num_groups=np.load(sbm_filepath+'_num_groups.npy',  allow_pickle=True )
        # ds.plot_SBM(matrix=sbm_matrix_arr[4], node_counts=new_node_counts[4])
        # ds.plot_entropy_groups(entropy_arr=sbm_entropy, groups_arr=sbm_num_groups)
        
        print('Loading Data')
        
        ds = ES_Graph_tool(nc_file=fname, name=name, es_network_file=networkfile_lb,
                            lon_range=lon_range,
                            lat_range=lat_range,
                           grid_step=grid_step)
        node_levels, sbm_node_counts=ds.node_level_arr(sbm_group_levels)
        ordered_node_levels=ds.parallel_ordered_nl_loc(node_levels)

        top_level=ordered_node_levels[-2]
        map=ds.get_map(top_level)
        savepath=PATH + f"/../../plots/asia/cluster_analysis/{job_id}_2_groups.png"
        ds.plot_map(map, dcbar=True, title=f"# {job_id}", projection='PlateCarree')
        plt.savefig(savepath)
    

    

  