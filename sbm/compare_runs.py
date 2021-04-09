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
from numpy.core.fromnumeric import _ravel_dispatcher
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../") # Adds higher directory 

from src.monsoon_region_es import Monsoon_Region_ES
from src.climnet import ClimNet
from src.dendrograms import Dendrogram_ES
from src.es_graph_tool import ES_Graph_tool
import src.link_bundles as lb
from src.dataset import BaseDataset
#%%
"""Class of monsoon regions, containing the
monsoon definitions, the node ids and the regional monsoons."""

class Compare_Runs(Monsoon_Region_ES):
    """ Dataset for analysing the network output and
    comparing multiple cluster runs from graph tool.

    Args:
    ----------
    nc_file: str  
        filename  
    var_name: str
        Variable name of interest
    """


    def __init__(self, nc_file, network_file=None, var_name=None,
                 group_levels=None, 
                 lon_range=[-180,180],
                 lat_range=[-90,90],
                 time_range=['1997-01-01', '2019-01-01'],
                 grid_step=1.0, name='pr', anomalies=False,lsm=False,
                 num_jobs=1, num_cpus=16, load=False):
        super().__init__(nc_file=nc_file, var_name=var_name, 
                 network_file=network_file,
                 time_range=time_range,
                 group_levels=group_levels,
                 lon_range=lon_range,
                 lat_range=lat_range,
                 grid_step=grid_step, name=name, anomalies=anomalies,lsm=lsm,
                 load=load
                 )
        self.PATH = os.path.dirname(os.path.abspath(__file__))

        if network_file is None:
            self.climnet=self.create_network(var_name, grid_step, 
                                             num_jobs=num_jobs, num_cpus=num_cpus)
        else:
            self.climnet=ClimNet(self, network_file=network_file)

        sparsity_adj = np.count_nonzero(self.climnet.adjacency.flatten())/self.climnet.adjacency.shape[0]**2
        print(f"Sparsity of corrected adjacency matrix: {sparsity_adj:.5f}")

        self.job_id=0




    def create_es_network(self,  vname, grid_step, linkbund=True, savenet=True, num_jobs=1,
                          num_cpus=16):
        print('Create and store Network')
        Network = ClimNet(self, corr_method='es', num_jobs=num_jobs)

        sparsity = np.count_nonzero(Network.adjacency.flatten())/Network.adjacency.shape[0]**2
        print(f"Sparsity of uncorrected adjacency matrix: {sparsity}")

        # store 
        if savenet == True:
            networkfile = self.PATH + f"/../outputs/{vname}_ES_net_{grid_step}.npz"
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
            sparsity_corr = np.count_nonzero(adjacency_corr.flatten())/adjacency_corr.shape[0]**2
            print(f"Sparsity of corrected adjacency matrix: {sparsity_corr}")
            if savenet == True:
                networkfile = self.PATH + f"/../outputs/{vname}_link_bundle_net_{grid_step}.npz"
                network_corr = Network.convert2sparse(adjacency_corr)
                Network.store_network(network_corr, networkfile)
        
        return Network

    ############ Cluster Analysis ############
    def compare_monsoon_arr(self, num_runs=10, m_arr=None,  scott_factor=1, plot=False, graph_file=None):
        if m_arr is None:
            raise ValueError(f"Please select array of monsoon keys of specific locations!")
        
        
        cluster_idx_arr=np.zeros((num_runs, len(self.indices_flat)))
        av_num_links=np.count_nonzero(self.climnet.adjacency)/self.climnet.adjacency.shape[0]

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

            # Maybe try different thresholds!
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
    
    def get_ids_loc_arr(self, loc_arr):
        sel_ids=[]
        for loc in loc_arr:
            sel_ids.append(self.get_n_ids(loc) )
        return sel_ids


    def get_density_cluster(self, loc_arr, num_runs=30,  abs_th=4, rel_th=0, 
                            graph_file=None):
        """
        This function returns the main cluster in a band like structure for selected lat.
        """
        
        cluster_idx_arr=np.zeros((num_runs, len(self.indices_flat)))

        sel_ids=self.get_ids_loc_arr(loc_arr=loc_arr)

        for idx, job_id in tqdm(enumerate(range(0,num_runs))):
            if graph_file is None:
                sbm_filepath=(self.PATH + 
                              f"/graphs/{self.dataset_name}_{self.grid_step}/{job_id}_{self.dataset_name}_graph_tool_ES_{self.grid_step}")
            else:
                sbm_filepath=self.PATH + f"{job_id}_" + graph_file

            if not os.path.exists(sbm_filepath +'_group_levels.npy'):
                print(f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
                continue
            group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )
            d_es=Dendrogram_ES(group_levels,)

            # Compute the cluster which is given by the returned leaf node ids
            leaf_nodes =self.get_cluster_sel_ids(sel_ids=sel_ids, d_es=d_es, abs_th=abs_th, rel_th=rel_th)
            
            if len(leaf_nodes) == len(self.indices_flat):
                print(f"JobID {job_id}: Warning all nodes in one cluster!")
            for id in np.concatenate(sel_ids):
                if id not in leaf_nodes:
                    print(f"JobId {job_id}: Warning monsoon id {id} not in cluster ids!")

            cluster_idx_arr[idx,:]=self.flat_idx_array(leaf_nodes)
            
        mean=np.mean(cluster_idx_arr, axis=0)
        std=np.std(cluster_idx_arr, axis=0)

        return mean, std
    
    def get_cluster_sel_ids(self, sel_ids, d_es=None, abs_th=0, rel_th=1):
        """
        Returns the node ids of the an array of selected node ids.
        """
        if d_es is None:
            d_es=self.d_es
        g_Zid=d_es.d_tree.get_split_groups(sel_ids, abs_th=abs_th, rel_th=rel_th )
        leaf_nodes= d_es.d_tree.get_leaf_nodes_of_is_node(g_Zid)
        
        return leaf_nodes


    def compare_grouping(self, loc, num_last_levels=5, num_runs=10, scott_factor=1,  graph_file=None):
        """
        Compares the groupings of group_levels for different runs
        
        Args
        ------
        loc: location as (lon, lat)
        """
        coord_deg, coord_rad, map_idx = self.get_coordinates_flatten()
        cluster_idx_arr=np.empty((num_last_levels, num_runs, coord_rad.shape[0]))
        av_num_links=np.count_nonzero(self.climnet.adjacency)/self.climnet.adjacency.shape[0]

        bw_opt= scott_factor * av_num_links**(-1./(2+4)) # Scott's rule of thumb
        for idx, job_id in enumerate(range(0,num_runs)):
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

            node_levels=d_es.node_levels
            # ordered_node_levels, nl_lat_dict=self.parallel_ordered_nl_loc(node_levels)
            num_levels=len(node_levels)
            for lidx, lid in enumerate(range(num_levels- num_last_levels, num_levels)):
                # lat_this_level=nl_lat_dict[lid]['lat']
                # lon_this_level=nl_lat_dict[lid]['lon']
                # loc_this_level=list(zip(lon_this_level, lat_this_level))
                # g_id, _= self.find_min_distance(loc_this_level, loc)
                # g_id,_=self.find_nearest(lat_this_level, loc)

                idx_loc=self.get_index_for_coord(lon=loc[0], lat=loc[1])
                g_id=node_levels[lid][idx_loc] # Get the group number in which the location occurs

                leaf_nodes= np.where(node_levels[lid]==g_id)[0]
                
                c_coord=coord_rad[leaf_nodes]
                occ=lb.spherical_kde(c_coord, coord_rad,bw_opt=bw_opt)
                den=occ/max(occ)
                cluster_idx_arr[lidx, idx,:]=den

        mean_arr=[]
        std_arr=[]
        for lidx, lid in enumerate(range(num_levels- num_last_levels, num_levels)):

            mean, std, perc90, perc95, perc99, perc995, perc999 = lb.compute_stats(runs=cluster_idx_arr[lidx])
            mean_arr.append(mean)
            std_arr.append(std)
        return mean_arr, std_arr


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
    
        ax.legend(loc='upper right', bbox_to_anchor=(.25,1), bbox_transform=ax.transAxes)
        ax1_2.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        
        # fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)



        if savepath is not None:
            print(f"Store files to: {savepath}")
            fig.savefig(savepath)

        return mean_entropy, std_entropy, mean_num_groups, std_num_groups


#%% 
if __name__ == "__main__":
    cluster = True
    grid_step = 2.5
    vname = 'pr'
    num_cpus = 64
    num_jobs=16
    job_id=8
    if os.getenv("HOME") =='/home/goswami/fstrnad80':
        dirname = "/home/goswami/fstrnad80/data/GPCP/"
    else:
        dirname = "/home/strnad/data/GPCP/"
        # dirname = "/home/jakob/climate_data/local/"
    fname = dirname +"gpcp_daily_1996_2020_2p5_new.nc4"


    # %%
    job_id=0
    sbm_filepath=PATH + f"/graphs/{job_id}_{vname}_graph_tool_ES_{grid_step}"
    network_file=PATH + f"/../outputs/{vname}_link_bundle_ES_net_{grid_step}.npz"

    group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )                       
    ds = Compare_Runs(fname, var_name=vname, 
                        network_file=network_file,
                        group_levels=group_levels,
                        time_range=['1997-01-01', '2019-01-01'],
                        grid_step=grid_step)
    monsoon_dict=ds.monsoon_dictionary

    # %%
    mean_entropy, std_entropy, mean_num_groups, std_num_groups= ds.compare_entropy(num_runs=30,max_num_levels=11 )

    
    

    
    

    # %%
    m_arr=['India South Asia','North Africa', 'North America', 'East Asia', 'NWP' ]
    m_arr=['Australia','South America', 'South Africa' ]

    sel='SH'
    mean_cluster, m_arr=ds.compare_cluster_runs(num_runs=30, m_arr=m_arr, plot=False, scott_factor=.05)

    savepath=PATH + f"/../plots/monsoon_cluster/density_cluster_monsoons_{m_arr}.pdf"
    ds.plot_map(ds.get_map(mean_cluster), label='Density of occurence', title=f'Density Plot for {m_arr}', color='Reds',
                vmin=0, vmax=None )
    plt.savefig(savepath)