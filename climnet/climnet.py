"""Climate network class."""

import sys,os
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../") # Adds higher directory 

import numpy as np
import xarray as xr
import scipy.stats as stat
import scipy.sparse as sparse
import scipy.special as special
import multiprocessing as mpi
import src.link_bundles as lb
import src.event_synchronization as es
import copy

class ClimNet:
    """ Climate Network class.
    Args:
    ----------
    dataset: BaseDataset object
        Dataset
    corr_method: str
        Correlation method of network ['spearman', 'pearson'], default: 'spearman'
    network_file: str

    # TODO:
        - flatten array once in dataset and call here once
    """

    def __init__(self, dataset, corr_method='spearman', 
                 threshold=0.5, confidence=0.95, test='onesided', tau_max=10, run_evs= False, num_jobs=1,
                 network_file=None):
        self.dataset = dataset
        if network_file is None:
            if corr_method=='es':
                self.q=self.dataset.q
                self.tau_max=tau_max
                self.min_evs=self.dataset.min_evs

                if run_evs is True:
                    self.E_matrix_folder = PATH + f'/E_matrix/{self.dataset.dataset_name}/'
                    self.null_model_folder=f'null_model/{self.dataset.dataset_name}/'
                    
                    self.adjacency=self.full_event_synchronization_run(E_matrix_folder=self.E_matrix_folder )

            else:
                if corr_method=='spearman':
                    self.corr, self.pvalue = self.calc_spearman(self.dataset, test=test)
                elif corr_method=='pearson':
                    self.corr, self.pvalue = self.calc_pearson(self.dataset)
                
                self.adjacency = self.get_adjacency(self.corr, self.pvalue,
                                                    threshold, confidence)
        else:
            self.network = sparse.load_npz(network_file)
            self.adjacency = np.array(self.network.todense())
            self.sparsity = np.count_nonzero(self.adjacency.flatten())/self.adjacency.shape[0]**2
            print(f"Sparsity of adjacency matrix: {self.sparsity}")



    def calc_spearman(self, dataset, test='onesided'):
        """Spearman correlation of the flattened and remove NaNs object.
        TODO: check dimension for spearman (Jakob)
        """
        data = dataset.flatten_array()
        print(data.shape)

        corr, pvalue_twosided = stat.spearmanr(data, axis=0, nan_policy='propagate')

        if test=='onesided':
            pvalue, self.zscore = self.onesided_test(corr)
        elif test=='twosided':
            pvalue = pvalue_twosided
        else:
            raise ValueError('Choosen test statisics does not exist. Choose "onesided" '
                              + 'or "twosided" test.')
        
        print(f"Created spearman correlation matrix of shape {np.shape(corr)}")
        return corr, pvalue
    

    def onesided_test(self, corr):
        """P-values of one sided t-test of spearman correlation.
        Following: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
        """
        n = corr.shape[0]
        f = np.arctanh(corr)
        zscore = np.sqrt((n-3)/1.06) * f
        pvalue = 1 - stat.norm.cdf(zscore)

        return pvalue, zscore

    

    def calc_pearson(self, dataset):
        """Pearson correlation of the flattened array."""
        data = dataset.flatten_array()
        print(data.shape)
        # Pearson correlation
        corr = np.corrcoef(data.T)
        assert  corr.shape[0] == data.shape[1]

        # get p-value matrix
        # TODO: Understand and check the implementation
        rf = corr[np.triu_indices(corr.shape[0], 1)]
        df = data.shape[1] - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pf = special.betainc(0.5 * df, 0.5, df / (df + ts))
        p = np.zeros(shape=corr.shape)
        p[np.triu_indices(p.shape[0], 1)] = pf
        p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
        p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])

        return corr, p


    def event_synchronization_run(self, E_matrix_folder=None, tau_max=10, min_sync_ev=1, null_model_file=None):
        # Test if ES data is computed
        if self.dataset.data_evs is None:
            raise ValueError("ERROR Event Synchronization data is not computed yet")
        
        #for job array
        try:
            min_job_id=int(os.environ['SLURM_ARRAY_TASK_MIN'])
            max_job_id=int(os.environ['SLURM_ARRAY_TASK_MAX'])
            job_id  =  int(os.environ['SLURM_ARRAY_TASK_ID'])
            num_jobs=int(os.environ['SLURM_ARRAY_TASK_COUNT'])
            # num_jobs=max_job_id-min_job_id +1
            # num_jobs=30

            print(f"job_id: {job_id}/{num_jobs}, Min Job ID: {min_job_id}, Max Job ID: {max_job_id}" )
        except KeyError:
            job_id= 0
            num_jobs= 1
            print("Not running with SLURM job arrays, but with manual id: ", job_id)

        event_series_matrix=self.dataset.flatten_array(dataarray=self.dataset.data_evs, check=False ).T

        if E_matrix_folder is None:
            E_matrix_folder=self.E_matrix_folder

        if not os.path.exists(E_matrix_folder):
            os.makedirs(E_matrix_folder)
        E_matrix_filename = f'E_matrix_{self.dataset.dataset_name}_q_{self.q}_taumax_{tau_max}_min_num_events_{self.min_evs}_jobid_{job_id}.npy'

        if not os.path.exists(null_model_file):
            raise ValueError("Null model path does not exist {null_model_file}! ")
        
        if not os.path.exists(null_model_file):
            raise ValueError(f'Null model path does not exist! {null_model_file}')
        null_model=np.load(null_model_file)
        print('Null models shape: ', null_model.shape)


        print(f'JobID {job_id}: Start comparing all time series with tau_max={tau_max}!')
        E_matrix_parallel=es.parallel_event_synchronization(event_series_matrix,
                                                            tau_max=tau_max,
                                                            min_num_sync_events=min_sync_ev,
                                                            job_id=job_id,
                                                            num_jobs=num_jobs,
                                                            savepath=E_matrix_folder+E_matrix_filename,
                                                            null_model=null_model)

        if num_jobs>1 and job_id>min_job_id:
            sys.exit(0)

        return E_matrix_filename
    
    def compute_es_null_model(self, n_pmts=3000, null_model_folder=None):
        from math import ceil

        time_steps, len_lat, len_lon=self.dataset.data_evs.shape
        max_num_events=ceil (time_steps*(1-self.q))
        null_model_filename=f'null_model_{self.dataset.dataset_name}_mnoe_{max_num_events}_tau_max_{self.tau_max}'

        if null_model_folder is None:
            if not os.path.exists(self.null_model_folder):
                os.makedirs(self.null_model_folder)
            
            savepath=self.null_model_folder + null_model_filename
        else:
            savepath=null_model_folder + null_model_filename

        # If folder is empty
        if not os.path.exists(savepath + '_threshold_05.npy'):
            p1,p2,p3,p4,p5= es.null_model_distribution(length_time_series=time_steps, 
                                                   min_num_events=self.min_evs, 
                                                   max_num_events=max_num_events,
                                                   num_permutations=n_pmts, 
                                                   savepath=savepath)
        
        return savepath
        

    def compute_es_adjacency(self, E_matrix_folder, num_jobs=1, full_E_matrix=False):
        if not os.path.exists(E_matrix_folder):
            raise ValueError(f"ERROR! The parallel ES is not computed yet!")
        
        E_matrix_fn=f'E_matrix_{self.dataset.dataset_name}_q_{self.q}_taumax_{self.tau_max}_min_num_events_{self.min_evs}'
        E_matrix_filename=E_matrix_folder+E_matrix_fn+ f'_jobid_'
        if full_E_matrix:
            savepath=E_matrix_folder+E_matrix_fn+'_full.npy'
            E_matrix_ij_full=es.construct_full_matrix_of_parts(num_jobs, filename=E_matrix_filename,
                                                    savepath=savepath)
       
        _, num_time_series=self.dataset.flatten_array(dataarray=self.dataset.data_evs ).shape
        adj_matrix_null_model= es.get_null_model_adj_matrix_from_E_files(E_matrix_folder, 
                                                                          num_time_series,
                                                                          savepath=None)
        
        
        return adj_matrix_null_model


    def full_event_synchronization_run(self,E_matrix_folder=None, null_model_file=None,es_run=False, 
                                       networkfile=None, networkfile_lb=None, c_adj=True,
                                       linkbund=True, num_permutations_lb=1000, savenet=True,num_jobs=1, num_cpus=16 ):

        """
        This function has to be called twice, once, to compute the exact numbers of synchronous
        events between two time series, second again to compute the adjacency matrix 
        and the link bundles. 
        Attention: The number of parrallel jobs that were used for the E_matrix needs to be 
        passed correctly to the function.
        """
        if E_matrix_folder is None:
            E_matrix_folder = PATH + f'/E_matrix/{self.dataset.dataset_name}_{self.dataset.grid_step}/'
        
        if null_model_file is None:
            null_model_file=self.compute_es_null_model()
            sys.exit()
        else:
            if not os.path.exists(null_model_file):
                raise ValueError(f'File {null_model_file} does not exist!')

        # Compute E_matrix
        if es_run is True:
            _, num_jobs= self.event_synchronization_run(E_matrix_folder=E_matrix_folder, null_model_file=null_model_file)
            return None
        else:
            # Compute Adjacency based on E_matrix
            if c_adj is True:
                self.adjacency=self.compute_es_adjacency(E_matrix_folder=E_matrix_folder, 
                                                    num_jobs=num_jobs)
                sparsity = np.count_nonzero(self.adjacency.flatten())/self.adjacency.shape[0]**2
                print(f"Sparsity of adjacency matrix: {sparsity}")

                if savenet == True:
                    if networkfile is None:
                        networkfile = PATH + f"/../outputs/{self.dataset.dataset_name}_ES_net_{self.dataset.grid_step}.npz"
                    
                    self.network = self.convert2sparse(self.adjacency)
                    self.store_network(self.network, networkfile)
            else:
                if not os.path.exists(networkfile):
                    raise ValueError(f"ERROR! This network file does not exist: {networkfile}!")
                self.network = sparse.load_npz(networkfile)
                self.adjacency = np.array(self.network.todense())

            if linkbund == True:
                print('Start linkbundles...')
                self.adjacency = self.link_bundles(
                    adjacency=self.adjacency,
                    confidence=0.99, num_rand_permutations=num_permutations_lb,
                    num_cpus=num_cpus
                )
                sparsity = np.count_nonzero(self.adjacency.flatten())/self.adjacency.shape[0]**2

                print(f"Sparsity of Link bundles corrected adjacency matrix: {sparsity}")
                if savenet == True:
                    if networkfile_lb is None:
                        networkfile_lb = PATH + f"/../outputs/{self.dataset.dataset_name}_ES_net_lb_{self.dataset.grid_step}.npz"
                    
                    network = self.convert2sparse(self.adjacency)
                    self.store_network(network, networkfile_lb)
            return self.adjacency

    def get_adjacency(self, corr, pvalue, threshold=0.5, confidence=0.95):
        """Create adjacency matrix from spearman correlation.

        Args:
        -----
        corr: np.ndarray (N x N)
            Spearman correlation matrix
        pvalue: np.ndarray (N x N)
            Pairwise pvalues of correlation matrix
        threshold: float
            Threshold to cut correlation
        confidence: float
            Confidence level

        Returns:
        --------
        adjacency: np.ndarray (N x N)
        """
        mask_confidence = np.where(pvalue <= (1-confidence), 1, 0)
        mask_threshold = np.where(corr > threshold, 1, 0)
        adjacency = mask_confidence * mask_threshold
        print("Created adjacency matrix.")

        return adjacency

        

    def check_densityApprox(self, adjacency, idx):

        coord_deg, coord_rad, map_idx = self.dataset.get_coordinates_flatten()
        all_link_idx_this_node = np.where(adjacency[idx, :] > 0)[0]
        link_coord = coord_rad[all_link_idx_this_node]
        bw_opt = 0.2 * len(all_link_idx_this_node)**(-1./(2+4)) # Scott's rule of thumb
        Z = lb.spherical_kde(link_coord, coord_rad, bw_opt)

        return {'z': Z, 'link_coord': link_coord, 'all_link': all_link_idx_this_node}


    def link_bundles(self, adjacency, confidence, num_rand_permutations=1000,
                     num_cpus=mpi.cpu_count()):
        """Significant test for adjacency. """
        # Get coordinates of all nodes
        coord_deg, coord_rad, map_idx = self.dataset.get_coordinates_flatten()

        # First compute Null Model of old adjacency matrix
        link_bundle_folder = PATH + f'/link_bundles/{self.dataset.dataset_name}_{self.dataset.grid_step}/'
        null_model_filename = f'link_bundle_null_model_{self.dataset.dataset_name}_{self.dataset.grid_step}'

        print("Start computing null model of link bundles!")
        lb.link_bundle_null_model(
            adjacency, coord_rad,
            link_bundle_folder=link_bundle_folder,
            filename=null_model_filename,
            num_rand_permutations=num_rand_permutations,
            num_cpus=num_cpus
        )

        # Now compute again adjacency corrected by the null model of the link bundles
        print("Now compute new adjacency matrix!")
        adjacency = lb.link_bundle_adj_matrix(
            adjacency, coord_rad, link_bundle_folder, null_model_filename, 
            scott_factor=0.2, perc=999, num_cpus=num_cpus
        )
        return adjacency
    

    def convert2sparse(self, adjacency):
        """Convert adjacency matrix to scipy.sparce matrix.

        Args:
        -----
        adjacency: np.ndarray (N x N)
            Adjacency matrix of the network

        Returns:
        --------
        network: np.sparse
            The network a
        """
        network = sparse.csc_matrix(adjacency)
        print("Converted adjacency matrix to sparce matrix.")
        return network


    def store_network(self, network, fname):
        """Store network to file."""
        if os.path.exists(fname):
            print("Warning File" + fname + " already exists! Copied as backup!")
            os.rename(fname, fname+'_bak')
        sparse.save_npz(fname, network)
        print(f"Network stored to {fname}!")
    

    def get_node_degree(self):
        node_degree = []
        for node in self.adjacency:
            node_degree.append(np.count_nonzero(node))
        
        return np.array(node_degree)

