#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:11:04 2020

@author: Felix Strnad
"""

import graph_tool.all as gt
import numpy as np
import sys, os
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pyunicorn as pu
from joblib import Parallel, delayed

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../") # Adds higher directory 

from src.dataset import BaseDataset
from src.climnet import ClimNet

#%%
""" Create a class for a graph tool object that is applicable on the precipitation ES dataset """

class ES_Graph_tool(BaseDataset):
    """ Dataset for Creating Clusters provided by the graph_tool package.
    """

    def __init__(self, nc_file, network_file, name=None, var_name='pr', 
                lon_range=[-180, 180],
                 lat_range=[-90,90],
                 time_range=['1997-01-01', '2019-01-01'], evs=True, 
                 grid_step=1.0, anomalies=False, lsm=False, load=False,rrevs=False):

        super().__init__(nc_file=nc_file, var_name=var_name, 
                        lon_range=lon_range,
                        lat_range=lat_range,
                        time_range=time_range,
                        grid_step=grid_step, 
                        name=name, 
                        anomalies=anomalies,
                        lsm=lsm,
                        evs=evs,
                        rrevs=rrevs,
                        load=load
                        )
        PATH = os.path.dirname(os.path.abspath(__file__))

        self.PrecipES_net=ClimNet(self, network_file=network_file)
        
        
        graph_path=PATH + f'/graphs/{name}_{self.grid_step}/{name}_es_graph_gt_{self.grid_step}.xml.gz'
        self.graph=self.construct_graph_from_adj_matrix(adj_matrix=self.PrecipES_net.adjacency,
                                                        savepath=graph_path )

        # Check if graph is consistent with 
        g_N= self.graph.num_vertices()
        if g_N != len(self.indices_flat):
            raise ValueError(f"Too many indices in graph: {g_N} vs {len(self.indices_flat)}!")

    def construct_graph_from_adj_matrix(self,adj_matrix, savepath=None):
        # Preprocessing
        
        # ensure square matrix
        M, N = adj_matrix.shape
        if os.path.isfile(savepath):
            print(f"File already exists! Take file {savepath}")
            g=gt.load_graph(savepath)
            return g
        else:
            g_folder_path=os.path.dirname(savepath)
            if not os.path.exists(g_folder_path):
                os.makedirs(g_folder_path)
        if M != N:
            raise ValueError("Adjacency must be square!")

        network=pu.Network(adjacency=self.PrecipES_net.adjacency)
        edge_list=network.graph.get_edgelist() # TODO Write own get_edgelist function

        B=len(edge_list)
        print(N, B)

        # We start with an empty, directed graph
        g = gt.Graph()
        # Add N nodes
        g.add_vertex(N)

        # Now iterate through adj matrix to construct the network (graph)
        for idx, edge in enumerate( tqdm(edge_list)):
            source, target=edge
            g.add_edge(g.vertex(source), g.vertex(target) )

        print("Finished creating graph")
        if savepath is not None:
            g.save(savepath)
        print("Finished creating graph! Summary:")
        print(g)

        return g

    def get_sbm_matrix(self, e):
        """
        This functions stores the Matrix of edge counts between groups for the SBM

        Parameters
        ----------
        e : graph
            graph of the hierarchical level.

        Returns
        -------
        np.array
            Array of 2d-Matrices for each level.

        """

        matrix=e.todense()
        M,N = matrix.shape
        if M!=N:
            print(f"Shape SB Matrix: {M},{N}")
            print(f"ERROR! SB matrix is not square! {M}, {N}")
            sys.exit(1)

        return np.array(matrix)
    
    def collect_node_membership(self,nodes):
        group_membership_nodes=[]
        for node in nodes:
            group_membership_nodes.append(node)

        return group_membership_nodes

    def apply_SBM(self,g, B_min=None, B_max=None, parallel=False, epsilon=1e-3, equilibrate=False, savepath=None):
        import time

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        start = time.time()
        
        print("Start computing SBM on graph...")
        if parallel is True:
            mcmc_args={'sequential': False, 'parallel': True}
        else:
            mcmc_args={}
        state = gt.minimize_nested_blockmodel_dl(g, deg_corr=True, B_min=B_min, B_max=B_max, 
                                                mcmc_args=mcmc_args,mcmc_equilibrate_args={'epsilon': epsilon},
                                                )
        print("Finished minimize Nested Blockmodel!")
        state.print_summary()

        S1 = state.entropy()

        # we will pad the hierarchy with another xx empty levels, to give it room to potentially increase
        # TODO maybe try out first equilibration and then minimize nested blockmodel!
        if equilibrate is True:
            state = state.copy(bs=state.get_bs() + [np.zeros(1)] * 4,
                            sampling = True)
            state = state.copy(bs=state.get_bs() , sampling = True)
            print("Do further MCMC sweeps...")
            for i in range(2000):
                ret = state.multiflip_mcmc_sweep(niter=20, beta=np.inf)

            # Now equilibrate
            print("Now equilibrate results...")
            # gt.mcmc_equilibrate(state, nbreaks=2, wait=100, mcmc_args=dict(niter=10))

        S2 = state.entropy()
        print(f" Finished MCMC search! Improvement: {S2-S1}")

        # Now we collect the marginal distribution for exactly 100,000 sweeps
        niter=1000
        num_levels=len(state.get_levels())
        N=g.num_vertices()+1
        E=g.num_edges()
        h = np.zeros((num_levels,N))

        def collect_num_groups(s):
            for l, sl in enumerate(s.get_levels()):
                B = sl.get_nonempty_B()
                h[l][B] += 1

        print(f"Sample from the posterior in {niter} samples!")
        # gt.mcmc_equilibrate(state, force_niter=niter, mcmc_args=dict(niter=10),
        #                 callback=collect_num_groups)
        print(f"Finished sampling from the posterior.")

        # gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))

        levels = state.get_levels()
        group_levels=[]

        # The hierarchical levels themselves are represented by individual BlockState instances
        # obtained via the get_levels() method:
        entropy_arr=[]
        num_groups_arr=[]
        sbm_matrix_arr=[]
        num_groups_new=0
        num_groups_old=N
        for s in levels:
            e_here= s.get_matrix()
            nodes=s.get_blocks()
            group_membership_nodes=self.collect_node_membership(nodes)
            num_groups_new= s.get_nonempty_B()

            if num_groups_new<num_groups_old:
                # Group numbers
                print(f"New number of groups: {num_groups_new} < previous: {num_groups_old}")
                group_levels.append(group_membership_nodes)
                # SBM Matrx
                sbm_matrix=self.get_sbm_matrix(e_here)
                sbm_matrix_arr.append(sbm_matrix)
                # Total number of groups
                num_groups_arr.append(num_groups_new)
                # Minimum description length
                entropy=s.entropy()
                entropy_arr.append(entropy)

                num_groups_old=num_groups_new

        state.print_summary()

        group_levels=np.array(group_levels,dtype=object)
        group_levels=self.reduce_group_levels(group_levels)
        ng_last_level=self.count_elements(group_levels[-1])
        ng=len(ng_last_level)
        # To conclude always with one cluster!
        if ng>1:
            group_levels=np.array(group_levels.tolist()+ [np.zeros(ng)])


        entropy_arr= np.array(entropy_arr)
        sbm_matrix_arr=np.array(sbm_matrix_arr,dtype=object)
        num_groups_arr=np.array(num_groups_arr)
        if savepath is not None:
            sbm_folder_path=os.path.dirname(savepath)
            if not os.path.exists(sbm_folder_path):
                os.makedirs(sbm_folder_path)
            
            print(f"Store files to: {savepath}")
            np.save(savepath+'_group_levels.npy', group_levels )
            np.save(savepath+'_sbm_matrix.npy', sbm_matrix_arr )
            np.save(savepath+'_entropy.npy', entropy_arr )
            np.save(savepath+'_num_groups.npy', num_groups_arr )
            np.save(savepath+'_group_distribution.npy', h )

        end = time.time()
        run_time=end - start
        print(f"Elapsed time for SBM: {run_time:.2f}")
        
        return state, group_levels, sbm_matrix_arr, entropy_arr,num_groups_arr, h


    ################### Groupings of nodes and clusters ############
    def count_elements(self,arr):
        unique, counts = np.unique(arr, return_counts=True)
        this_arr_count=dict(zip(unique, counts))
        return this_arr_count

    def node_level_arr(self,level_arr):
        buff = np.array(level_arr[0])
        x = [buff]
        count_nodes=[self.count_elements(buff)]
        for l in range(1,len(level_arr)):
            buff = np.array(level_arr[l])[buff]
            this_node_count=self.count_elements(buff)
            x.append(buff)
            count_nodes.append(this_node_count)

        return np.array(x), count_nodes

    def level_dict(self, arr_levels):
        """
        Gives dict for which nodes before are merged into this node.
        Works for group_levels and node_levels
        """
        level_dict=dict()
        for l_id, level_ids in enumerate(arr_levels):
            this_node_count=self.count_elements(level_ids)
            level_dict[l_id]=this_node_count
        return level_dict

    def node_level_dict(self, node_levels):
        """
        Gives for each level, for each group number which leaf nodes are in it.
        """
        node_level_dict=dict()
        for lid, level_ids in enumerate(node_levels):
            group_ids=np.unique(level_ids)
            this_level=[]                
            for idx, gid in enumerate(group_ids):
                node_idx=np.where(level_ids==gid)[0]
                if idx!= int(gid):
                    raise ValueError(f"Attention group ID missing: {gid} for idx {idx}!")
                this_level.append(node_idx)
            node_level_dict[lid]=this_level

        return node_level_dict

    def reduce_node_levels(self,node_levels):
        """
        Graph_tool with MCMC search does sometimes skip certain group numbers.
        This function brings back the ordering to numbers from 0 to len(level).
        """
        red_hierach_data=[]
        trans_dict=dict()
        level_dict=self.level_dict(node_levels)
        node_red_dict=dict()
        for l_id, this_level_dict in enumerate(level_dict.values()):
            this_trans_dict=dict()
            for i, (group_id, group_count) in enumerate(this_level_dict.items()):
                this_trans_dict[group_id]=i
            trans_dict[l_id]=this_trans_dict

        for l_id, level_ids in enumerate(node_levels):
            this_level=[]
            for level_id in level_ids:
                this_level.append(trans_dict[l_id][level_id])
                if l_id==0:
                    node_red_dict[level_id]=trans_dict[l_id][level_id]
            red_hierach_data.append(this_level)

        return np.array(red_hierach_data)

    def get_upper_level_group_number(self,arr):
        """
        Returns for one level which groups belong to which group number
        in the upper level.
        """
        unique_sorted=np.unique(arr)
        orig_dict= dict()
        for i in unique_sorted:
            occ_in_arr=np.where(arr==i)
            orig_dict[i]=occ_in_arr[0]
        return orig_dict

    def get_hierarchical_data_from_nodes(self,node_through_levels):
        new_hierarchical_data=[]
        node_group_dict=dict()
        for l_id, level in enumerate(node_through_levels):
            upper_level_groups=self.get_upper_level_group_number(level)
            this_level_arr=np.zeros(len(level), dtype=int)
            if l_id==0:
                for (group_id, group_count) in upper_level_groups.items() :
                    for i in group_count:
                        this_level_arr[i]=group_id
                        node_group_dict[i]=group_id
            else:
                lower_level=self.get_upper_level_group_number(node_through_levels[l_id-1])
                this_level_arr=np.zeros(len(lower_level), dtype=int)

                for (group_id, group_count) in upper_level_groups.items() :
                    for i in group_count:
                        this_group=node_group_dict[i]
                        this_level_arr[this_group]=group_id
                        node_group_dict[i]=group_id
            new_hierarchical_data.append(this_level_arr)

        return np.array(new_hierarchical_data, dtype=object)

    def reduce_group_levels(self,group_levels):
        node_levels,_=self.node_level_arr(group_levels)
        new_node_levels =self.reduce_node_levels(node_levels)
        red_group_levels=self.get_hierarchical_data_from_nodes(new_node_levels)

        return red_group_levels

    def foreward_node_levels(self, node_levels, node_id):
        foreward_path=[]
        for l in node_levels:
            group_num=l[node_id]
            foreward_path.append(group_num)
        return np.array(foreward_path)

    def get_sorted_loc_gid(self, group_level_ids, lid):
        mean_lat_arr=[]
        mean_lon_arr=[]
        result_dict=dict()
        for gid, node_ids in enumerate(group_level_ids):
            lon_arr=[]
            lat_arr=[]
            for nid in node_ids:
                map_idx=self.get_map_index(nid)
                lon_arr.append(map_idx['lon'])
                lat_arr.append(map_idx['lat'])
            # loc_dict[gid]=[np.mean(lat_arr), np.mean(lon_arr)]
            mean_lat_arr.append(np.mean(lat_arr))
            mean_lon_arr.append(np.mean(lon_arr))
        # sorted_ids=np.argsort(mean_lat_arr)  # sort by arg
        sorted_ids= [sorted(mean_lat_arr).index(i) for i in mean_lat_arr] # relative sorting
        if len(sorted_ids)!=len(mean_lat_arr):
            raise ValueError("Error! two lats with the exact same mean!")
        sorted_lat=np.sort(mean_lat_arr)  # sort by latitude
        sorted_lon=np.array(mean_lon_arr)[sorted_ids]
        result_dict['lid']=lid
        result_dict['sorted_ids']=sorted_ids
        result_dict['sorted_lat']=sorted_lat
        result_dict['sorted_lon']=sorted_lon
        return result_dict

    def parallel_ordered_nl_loc(self, node_levels):
        import multiprocessing as mpi

        nl_dict=self.node_level_dict(node_levels)
        new_node_levels=np.empty_like(node_levels)
        
        # For parallel Programming
        num_cpus_avail=mpi.cpu_count()
        print(f"Number of available CPUs: {num_cpus_avail}")
        backend='multiprocessing'
        # backend='loky'
        #backend='threading'
        parallelSortedLoc= (Parallel(n_jobs=num_cpus_avail,backend=backend)
                        (delayed(self.get_sorted_loc_gid)
                        (group_level_ids, lid)
                        for lid, group_level_ids in tqdm(nl_dict.items())  )
                        )
        loc_dict=dict()
        
        for result_dict in parallelSortedLoc:
            lid=result_dict['lid']
            sorted_ids=result_dict['sorted_ids']
            loc_dict[lid]={'lat':np.array(result_dict['sorted_lat']),'lon':np.array(result_dict['sorted_lon'])}
            for gid, node_ids in enumerate(nl_dict[lid]):
                new_node_levels[lid][node_ids]=sorted_ids[gid]

        return new_node_levels, loc_dict

    def ordered_nl_loc(self, node_levels):
        nl_dict=self.node_level_dict(node_levels)
        new_node_levels=np.empty_like(node_levels)
        for lid, group_level_ids in tqdm(nl_dict.items()):
            res_sorted_dict=self.get_sorted_loc_gid(group_level_ids, lid)
            sorted_loc=res_sorted_dict['sorted_loc']  # sort by latitude
            for gid, node_ids in enumerate(group_level_ids):
                new_node_levels[lid][node_ids]=sorted_loc[gid]
        
        return new_node_levels

    ############ Plotting!#################
    def compute_link_density(self, n, num_edges):
        pot_con=n*(n-1)/2  # Kleiner Gauss
        den=num_edges/pot_con
        return den
    
    def plot_SBM(self, matrix, node_counts, title='', vmax=None, savepath=None):
        """
        Plot SBM. The colums describe the nodes, the rows the connections to which node!

        Parameters
        ----------
        matrix : np.array
            input 2D matrix contains the edge counts. The default is None.
        node_counts : dict
            Contains the dictionary of number of nodes that contains each group.

        Returns
        -------
        density_matrix : np.array
            Density Matrix of SBM, same shape as matrix
        matrix : np.array
            same as input matix to allow detailed comparison.

        """
        import matplotlib
        matplotlib.rc('text', usetex=True)
        from matplotlib import ticker
        M,N=matrix.shape
        if M!=N:
            print(f"ERROR! SBM Matrix not squre {M} != {N} !")
            sys.exit(1)

        red_M=max(node_counts)+1
        extent=[0,red_M, 0,red_M]
        print(f"Shape={M}, {M}")

        density_matrix=np.zeros((M, M))

        for row_idx, row in enumerate(matrix):
            for group_idx, ec in enumerate(row):
                if group_idx in node_counts:
                    N=node_counts[group_idx]
                    density_matrix[row_idx][group_idx]=self.compute_link_density(N, ec)


        fig, ax = plt.subplots(figsize=(8,8))
        # Be sure to only pick integer tick locations.
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.imshow(density_matrix[::-1,:], vmax=vmax, interpolation='nearest', cmap='Greys', 
                            extent=extent)
        plt.colorbar(orientation='vertical', shrink=0.8, label=rf"$P(b_i\rightarrow b_j)$")
        ax.set_xlabel("Group Number")
        ax.set_ylabel("To Group number")
        ax.set_title(title, y=1.02)
        fig.tight_layout()
        if savepath is not None:
            print(f"Store SBM plot to: {savepath}")
            plt.savefig(savepath)

        return density_matrix, matrix

    def plot_entropy_groups(self,entropy_arr,groups_arr, savepath=None):
        fig, ax = plt.subplots(figsize=(7,7))
        num_levels=len(entropy_arr)
        ax.set_xlabel('Level')

        # Entropy
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(rf'Description Length $\Gamma$')
        ax.set_yscale('linear')
        plt_E=ax.plot(np.arange(1,num_levels+1), np.log(entropy_arr), color='tab:blue', label='Entropy')

        # Number of Groups
        ax1_2=ax.twinx()
        ax1_2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1_2.set_ylabel('Number of groups [1]')
        ax1_2.set_yscale('log')
        plt_G=ax1_2.plot(np.arange(1,num_levels+1), groups_arr, color='tab:green', label='Groups')

        plts=plt_E + plt_G
        labs = [l.get_label() for l in plts]

        ax.legend(plts, labs,loc='upper right',  fancybox=True)
        if savepath is not None:
            print(f"Store files to: {savepath}")
            fig.savefig(savepath)

        return None


# %%    
if __name__ == "__main__":
    grid_step = 2.5
    vname = 'pr'
    num_cpus = 64

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
        job_id=4
        num_jobs= 7


    if os.getenv("HOME") =='/home/goswami/fstrnad80':
        dirname = "/home/goswami/fstrnad80/data/GPCP/"
    else:
        dirname = "/home/strnad/data/GPCP/"
        # dirname = "/home/jakob/climate_data/local/"

    network_file=PATH + f"/../outputs/{vname}_link_bundle_ES_net_{grid_step}.npz"
    print('Loading Data')
    ds = ES_Graph_tool( name=vname, network_file=network_file,
                        grid_step=grid_step)

    # %%
    sbm_filepath=PATH + f"/graphs/{job_id}_{vname}_graph_tool_ES_{grid_step}"
    # %%
    state, group_levels, sbm_matrix_arr, entropy_arr,num_groups_arr, h= ds.apply_SBM(g=ds.graph, savepath=sbm_filepath)


    # %%
    job_id=3
    sbm_filepath=PATH + f"/graphs/{job_id}_{vname}_graph_tool_ES_{grid_step}"
    if not os.path.exists(sbm_filepath +'_group_levels.npy'):
        print(f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
    sbm_matrix_arr=np.load(sbm_filepath+'_sbm_matrix.npy',  allow_pickle=True )
    sbm_entropy=np.load(sbm_filepath+'_entropy.npy',  allow_pickle=True )
    sbm_group_levels=np.load(sbm_filepath+'_group_levels.npy',  allow_pickle=True )
    sbm_num_groups=np.load(sbm_filepath+'_num_groups.npy',  allow_pickle=True )
    
    print('Loading Data')
    ng_last_level=np.unique(sbm_group_levels[-1])
    ng=len(ng_last_level)
    if ng>1:
        print(f"WARNING! Last level group levels not correct! NG: {ng}!")

    
    node_levels, sbm_node_counts=ds.node_level_arr(sbm_group_levels)
    
    ds.plot_entropy_groups(entropy_arr=sbm_entropy, groups_arr=sbm_num_groups)

    # %%
    parallel_ordered_node_levels=ds.parallel_ordered_nl_loc(node_levels)
    # %%
    ordered_node_levels=ds.ordered_nl_loc(node_levels)
    # %%
    # Visualize Clustering
    
    for idx, job_id in enumerate(range(0,30)):
        print("Start Reading data jobid {job_id}...")
        sbm_filepath=PATH + f"/graphs/{job_id}_{vname}_graph_tool_ES_{grid_step}"
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
        ng_last_level=np.unique(sbm_group_levels[-1])
        ng=len(ng_last_level)
        if ng>1:
            print(f"WARNING! Last level group levels not correct! NG: {ng}!")
            continue

        ds = ES_Graph_tool(var_name=vname, network_file=network_file,
                           grid_step=grid_step)
        node_levels, sbm_node_counts=ds.node_level_arr(sbm_group_levels)
        ordered_node_levels=ds.parallel_ordered_nl_loc(node_levels)

        top_level=ordered_node_levels[-2]
        map=ds.get_map(top_level)
        savepath=PATH + f"/../plots/cluster_analysis/{job_id}_2_groups.png"
        ds.plot_map(map, dcbar=True, title=f"# {job_id}")
        plt.savefig(savepath)

    



    # %% 
    # Map
    # red_node_levels=ds.reduce_node_levels(sbm_node_levels)
    # top_level=sbm_node_levels[-2]
    # map=ds.get_map(top_level)
    # ds.plot_map(map, dcbar=True)



# %%


# group_levels=sbm_group_levels[-3:]
# node_levels,_=ds.node_level_arr(group_levels)
# new_node_levels =ds.reduce_node_levels(node_levels)

# red_group_levels=ds.get_hierarchical_data_from_nodes(new_node_levels)
# print(group_levels)
# print(red_group_levels)



# print(node_levels)
# print(new_node_levels)
# %%
