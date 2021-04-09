""" Base class for the different dataset classes of the multilayer climate network."""

import sys,os
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../") # Adds higher directory 

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.style.use('./src/matplotlib_style.py')
from cartopy import config
import cartopy.crs as ccrs
import cartopy as ctp

def load_dataset(filename):
    """"""
    ds = xr.open_dataset(filename)
    names=[]
    for name, da in ds.data_vars.items():
        print("Dataarray name: ", name)
        names.append(name)

    dataset = BaseDataset(
        filename, 
        var_name = names[0],
        time_range = [da.time.data[0], da.time.data[-1]],
        lon_range = [da.lon.min(), da.lon.max()],
        lat_range = [da.lat.min(), da.lat.max()],
        grid_step=ds.attrs['grid_step'],
        grid_type = ds.attrs['grid_type'],
        anomalies = bool(ds.attrs['anomalies']),
        lsm = bool(ds.attrs['lsm']),
        evs= bool(ds.attrs['evs']),
        load=True
    )
    return dataset
    
    
class BaseDataset:
    """ Base Dataset.
    Args:
    
    ----------
    nc_file: str
        filename of anomaly data
    var_name: str
        Variable name of interest
    lon_range: list [min, max]
        range of longitudes
    lat_range: list [min, max]
        range of latitudes
    """

    def __init__(self, nc_file, 
                 var_name=None,
                 time_range=['1997-01-01', '2019-01-01'],
                 lon_range=[-180, 180],
                 lat_range=[-90,90],
                 grid_step=1,
                 grid_type='gaussian',
                 name=None,
                 anomalies=False,
                 start_month=None,
                 end_month=None,
                 lsm=False,
                 evs=False,
                 rrevs=False,
                 load=False
                 ):

        if not os.path.exists(nc_file):
            PATH = os.path.dirname(os.path.abspath(__file__))
            print(f"You are here: {PATH}!")
            raise ValueError(f"File does not exist {nc_file}!")

        print(f"Load dataset: {nc_file}")
        ds = xr.open_dataset(nc_file)
        ds = self.check_dimensions(ds)
        
        self.time_range = time_range
        self.grid_step = grid_step
        self.grid_type = grid_type
        self.anomalies = anomalies
        self.lsm = lsm
        self.months=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        self.dataset_name = name
        if self.anomalies is True:
            self.dataset_name+='_anomalies'
        self.evs=evs
        
        # Event synchronization
        self.q=0.95
        self.min_evs=20
        self.min_treshold=1
        self.th_eev=15
        self.mask=None
        # initialized dataset 
        if load is False:
            self.var_name = var_name
            
            da = ds[self.var_name]
            del ds
            # Bring always in the form (time, lat, lon)
            da=da.transpose('time', 'lat', 'lon')  # much less memory consuming than for dataset!
            # choose time range
            da, self.time_grid = self.common_times(da, self.time_range)
            if start_month is not None and end_month is not None:
                da = self.get_month_range_data(da, start_month=start_month, end_month=end_month)

            # If lon from 0 to 360 shift to -180 to 180
            if max(da.lon)>180:
                print("Shift longitude!")
                da=da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
            # regridding
            self.lon_range = lon_range
            self.lat_range = lat_range
            
            # Cut map after common grid, to make it independent of input format!
            if self.lon_range!=[-180, 180] and self.lat_range!=[-90,90]:
                da=self.cut_map(da, self.lon_range, self.lat_range)

            da, self.grid, self.lat, self.lon = self.common_grid(da)
            
            
            if self.lsm is True:
                self.mask, da = self.get_land_sea_mask_data(da)

            # compute anomalies if not given in nc file
            if self.anomalies is True:
                print("WARNING! Anomalies are true!")
                self.data_anomalies = self.compute_anomalies(da, group='dayofyear')

            self.dataarray = da

            if self.evs is True:
            
                self.data_evs=self.create_event_time_series(th=self.min_treshold, 
                                                        q=self.q, 
                                                        min_evs=self.min_evs, 
                                                        th_eev=self.th_eev)
            
        # load dataset object from file
        else:
            
            names=[]
            for name, da in ds.data_vars.items():
                print("Dataarray name: ", name)
                names.append(name)
            self.var_name=names[0]
            da = ds[self.var_name]

            self.time_range = [da.time.data[0], da.time.data[-1]]
            self.lon_range = [float(da.lon.min()), float(da.lon.max()) ]
            self.lat_range = [float(da.lat.min()), float(da.lat.max()) ]
            self.grid_step=ds.attrs['grid_step']
            self.grid_type = ds.attrs['grid_type']
            self.anomalies = bool(ds.attrs['anomalies'])
            self.lsm = bool(ds.attrs['lsm'])
            self.evs= bool(ds.attrs['evs'])

            if self.anomalies is True:
                self.data_anomalies= ds['anomalies']
            if self.evs is True:
                self.data_evs=ds['evs']



            del ds
            self.time_grid = da.time
            self.grid = dict(lat=da.lat.data, lon=da.lon.data)
            self.dataarray = da
            
            # points which are always NaN will be NaNs in mask
            mask = np.ones_like(da[0].data, dtype=bool)
            
            if self.evs is True:
                da=self.data_evs
                for idx, t in enumerate(da.time):
                    mask *= np.where(da.sel(time=t).data==0, True, False)

                self.mask = xr.DataArray(
                                data=xr.where(mask==False, 1, np.NaN),
                                dims=da.sel(time=da.time[0]).dims,
                                coords=da.sel(time=da.time[0]).coords,
                                name='lsm')

            elif self.lsm is True:
                for idx, t in enumerate(da.time):
                    mask *= np.isnan(da.sel(time=t).data)

                self.mask = xr.DataArray(
                            data=xr.where(mask==False, 1, np.NaN),
                            dims=da.sel(time=da.time[0]).dims,
                            coords=da.sel(time=da.time[0]).coords,
                            name='lsm')

            
            
            if rrevs is True:
                print("Re-compute the event series again!")
                self.data_evs=self.create_event_time_series(th=self.min_treshold, 
                                                        q=self.q, 
                                                        min_evs=self.min_evs, 
                                                        th_eev=self.th_eev)



        # Flatten index in map
        if self.mask is not None:
            self.indices_flat, self.idx_map = self.init_mask_idx()

    def check_dimensions(self, ds):
        """
        Checks whether the dimensions are the correct ones for xarray!
        """
        lon_lat_names=['longitude', 'latitude']
        xr_lon_lat_names=['lon', 'lat']
        dims=list(ds.dims)
            
        for idx, lon_lat in enumerate(lon_lat_names ):
            if lon_lat in dims:
                print(dims)
                print(f'Rename:{lon_lat} : {xr_lon_lat_names[idx]} ')
                ds=ds.rename({lon_lat : xr_lon_lat_names[idx]})
                dims=list(ds.dims)
                print(dims)
        # if len(dims) != 3:     
        #     raise ValueError(f"Wrong dimension of input array for {dims}! It must include time, lon, lat")
        clim_dims=['time', 'lat', 'lon']
        for dim in clim_dims:
            if dim not in dims:
                raise ValueError(f"The dimension {dim} not consistent with required dims {clim_dims}!")
        return ds

    def common_grid(self, dataarray):
        """Common grid for all datasets.
        """
        
        min_lon=min(dataarray.lon)
        max_lon=max(dataarray.lon)
        min_lat=min(dataarray.lat)
        max_lat=max(dataarray.lat)
        init_lat = np.arange(min_lat, max_lat, self.grid_step)
        init_lon = np.arange(min_lon, max_lon, self.grid_step)
        grid = {'lat': init_lat, 'lon': init_lon}
        
        print(f"Interpolte grid from {float (min_lon)} to {float(max_lon)}, {float(min_lat)} to {float(max_lat)}!")
        da = dataarray.interp(grid, method='nearest')
        del dataarray
        return da, grid, init_lat, init_lon
    
    def cut_map(self, dataarray, lon_range, lat_range):
        """Cut an area in the map.

        Args:
        ----------
        lon_range: list [min, max]
            range of longitudes
        lat_range: list [min, max]
            range of latitudes

        Return:
        -------
        ds_area: xr.dataset
            Dataset cut to range
        """
        print(f"Cut Map: Lon Range:{min(lon_range)}, {max(lon_range)}, Lat Range: {min(lat_range)}, {max(lat_range)}!")
        lats=dataarray.lat.data
        if lats[0] > lats[-1]:
            ds_cut = dataarray.sel(
                lon=slice(np.min(lon_range), np.max(lon_range)),
                lat=slice(np.max(lat_range), np.min(lat_range)) # if lats are ordered differently go from + to minus!
            )
        else:    
            ds_cut = dataarray.sel(
                lon=slice(np.min(lon_range), np.max(lon_range)),
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
        return ds_cut

    def get_land_sea_mask_data(self, dataarray):
        """
        Compute a land-sea-mask for the dataarray,
        based on an input file for the land-sea-mask.
        """
        PATH = os.path.dirname(os.path.abspath(__file__)) # Adds higher directory 
        lsm_mask_ds = xr.open_dataset(PATH + "/../input/land-sea-mask_era5.nc")
        lsm_mask, lsm_grid, lsm_lat, lsm_lon = self.common_grid(lsm_mask_ds['lsm'])

        land_dataarray = xr.where(lsm_mask==1, dataarray, np.nan)
        return lsm_mask, land_dataarray

    def compute_anomalies(self, dataarray, group='dayofyear'):
        """Calculate anomalies.
        TODO: detrending?

        Args:
        -----
        group: str
            time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'

        Return:
        -------
        anomalies: xr.dataarray
        """
        anomalies = (dataarray.groupby(f"time.{group}")
                    - dataarray.groupby(f"time.{group}").mean("time"))
        print(f'Created {group}ly anomalies!')
        anomalies=anomalies.rename('anomalies')

        return anomalies


    def flatten_array(self, dataarray=None, time=True, check=True):
        """Flatten and remove NaNs.
        
        # TODO: self.dataarray should be only dataarray over land
        """
        if dataarray is None:
            dataarray=self.dataarray

        idx_land = np.where(self.mask.data.flatten() == 1)[0]
        if time is False:
            buff = dataarray.data.flatten()
            buff[np.isnan(buff)] = 0.0 # set missing data to climatology
            data= buff[idx_land]
        else:
            data = []
            for idx, t in enumerate(dataarray.time):
                buff = dataarray.sel(time=t.data).data.flatten()
                buff[np.isnan(buff)] = 0.0 # set missing data to climatology
                data.append(buff[idx_land])
        
        # check
        if check is True:
            num_nonzeros = np.count_nonzero(data[-1])
            num_landpoints = sum(~np.isnan(self.mask.data.flatten()))
            print(f"The number of non-zero datapoints {num_nonzeros} "
                + f"should approx. be {num_landpoints}.")

        return np.array(data)


    def save_to_file(self, filepath):
        """Save dataset or dataarray to file."""
        overwrite='y'
        try:
            if os.path.exists(filepath):
                print("File" + filepath + " already exists!")
                overwrite = input('File already exists. Overwrite? Y = yes, N = no\n')
            if overwrite.lower() == 'y':
                self.dataarray.to_netcdf(filepath)
                print(f"Stored dataset to {filepath}")
        except OSError:
                print("Could not write to file!")
        return

    def init_mask_idx(self):
        """
        Initializes the flat indices of the map.
        Usefule if get_map_index is called multiple times.
        """
        mask_arr = np.where(self.mask.data.flatten()==1, True, False)
        indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

        idx_map = self.get_map(indices_flat, name='idx_flat')
        return indices_flat, idx_map

    def get_map(self, data, name=None):
        """Restore dataarray map from flattened array.

        TODO: So far only a map at one time works, extend to more than one time

        This also includes adding NaNs which have been removed.
        Args:
        -----
        data: np.ndarray (n,0)
            flatten datapoints without NaNs
        mask_nan: xr.dataarray
            Mask of original dataarray containing True for position of NaNs
        name: str
            naming of xr.DataArray

        Return:
        -------
        dmap: xr.dataArray
            Map of data
        """
        mask_arr = np.where(self.mask.data.flatten()==1, True, False)
        # Number of non-NaNs should be equal to length of data
        assert np.count_nonzero(mask_arr) == len(data)

        # create array with NaNs
        data_map = np.empty(len(mask_arr)) 
        data_map[:] = np.NaN

        # fill array with sample
        data_map[mask_arr] = data

        dmap = xr.DataArray(
            data=np.reshape(data_map, self.mask.data.shape),
            # coords=[self.dataarray.coords['lat'], self.dataarray.coords['lon']],
            coords= self.mask.coords,
            name=name)

        return dmap
    
    def get_dataarray_idx(self, dataarray, idx_lst):
        """
        For a dataarray returns an xarray dataarray of only the selected indices of the flattend array,
        i.e. it takes into account the mask.
        """
        idx_map=self.get_map(self.flat_idx_array(idx_lst))

        idx_dataarray=xr.where(idx_map==1, dataarray, np.nan)
        return idx_dataarray

    def get_map_index(self, idx_flat):
        """Get lat, lon and index of map from index of fatten array 
           without Nans.

        # Attention: Mask has to be initialised
        
        Args:
        -----
        idx_flat: int, list
            index or list of indices of the flatten array with removed NaNs
        
        Return:
        idx_map: dict
            Corresponding indices of the map as well as lat and lon coordinates
        """

        indices_flat = self.indices_flat
        if idx_flat > len(indices_flat):
            raise ValueError(f"Index {idx_flat} doesn't exist.")
            
        idx_map = self.idx_map

        buff = idx_map.where(idx_map == idx_flat, drop=True)
        
        map_idx = {
            'lat': buff.lat.data,
            'lon': buff.lon.data,
            'idx': np.argwhere(idx_map.data == idx_flat)
        }
        return map_idx
    
    def get_loc_arr_for_index(self, idx_lst):
        """
        Returns a location array for a list of indices with already removed nans.
        """
        loc_arr=[]
        for idx in idx_lst:
            map_idx=self.get_map_index(idx)
            loc_arr.append([map_idx['lon'], map_idx['lat']])
        return np.array(loc_arr)

    def get_coordinates_flatten(self, dataarray=None):
        """Get coordinates of flatten array with removed NaNs.

        Return:
        -------
        coord_deg:
        coord_rad: 
        map_idx:
        """
        # length of the flatten array with NaNs removed
        length = self.flatten_array(dataarray=dataarray).shape[1]

        coord_deg = []
        map_idx = []
        for i in range(length):
            buff = self.get_map_index(i)
            coord_deg.append([buff['lat'][0], buff['lon'][0]])
            map_idx.append(buff['idx'][0])

        coord_rad = np.radians(coord_deg)

        return np.array(coord_deg), coord_rad, map_idx

    def get_index_for_coord(self,lon, lat):
        mask_arr = np.where(self.mask.data.flatten()==1, True, False)
        indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

        idx_map = self.get_map(indices_flat, name='idx_flat')

        idx=idx_map.sel(lat=lat, lon=lon, method='nearest')
        if np.isnan(idx):
            print("Warning the lon lat is not defined!")
        return int(idx)

    def flat_idx_array(self, idx_list):
        """
        Returns a flattened list of indices where the idx_list is at the correct position.
        """
        mask_arr = np.where(self.mask.data.flatten()==1, True, False)
        len_index=np.count_nonzero(mask_arr)
        full_idx_lst=np.zeros(len_index)
        full_idx_lst[idx_list]=1
        
        return full_idx_lst

    def find_nearest(self,a, a0):
        """
        Element in nd array `a` closest to the scalar value `a0`
        ----
        Args a: nd array
             a0: scalar value
        Return
            idx, value
        """
        idx = np.abs(a - a0).argmin()
        return idx, a.flat[idx]

    def haversine(self, lon1, lat1, lon2, lat2, r=1):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        # r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def find_min_distance(self, loc_arr, loc):
        """
        Finds the minimum distance between an array of lon,lat locations 
        and a specified location (lon,lat)
        """
        lon2,lat2=loc
        dist_arr=[]
        for (lon_c, lat_c) in loc_arr:
            dist_arr.append(self.haversine(lon_c, lat_c, lon2,lat2))
        
        return np.argmin(dist_arr), min(dist_arr)

    def common_times(self, data, time_range):
        """Common time steps.
        """
        t = np.arange(
            time_range[0], time_range[1], dtype='datetime64[D]'
        )
        time_grid = {'time': t}
        td=data.time.data
        if (td[0] > np.datetime64(time_range[0])) or (td[-1] < np.datetime64(time_range[1]) ) : 
            raise ValueError(f"Please select time array within {td[0]} - {td[-1]}!")
        else:
            print(f"Time steps within {time_range} selected!")
        # da = data.interp(time_grid, method='nearest')
        da=data.sel(time=slice(time_range[0], time_range[1]))
        return da, time_grid

    def get_data_timerange(self, data, time_range):
        start_time, end_time=time_range
        print(f"Get data within timerange: {start_time} - {end_time}")

        dataarray=data.sel(time=slice(start_time, end_time))
        print("Done!")

        return dataarray

    def _get_index_of_month(self,month):
        idx=-1
        idx=self.months.index(month)
        if idx==-1:
            print("This month does not exist: ", month)
            sys.exit(1)
        return idx


    def _is_in_month_range(self,month, start_month, end_month):
        start_month_idx=self._get_index_of_month(start_month)+1
        end_month_idx  =self._get_index_of_month(end_month)+1

        if start_month_idx<=end_month_idx:
            mask= (month >= start_month_idx) & (month <= end_month_idx)
        else:
            mask= (month >= start_month_idx) | (month <= end_month_idx)
        return mask


    def get_month_range_data(self, dataarray, start_month='Jan', end_month='Dec'):
        """
        This function generates data within a given month range.
        It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
        to smaller month (eg. Dec-Feb)

        Parameters
        ----------
        start_month : string, optional
            Start month. The default is 'Jan'.
        end_month : string, optional
            End Month. The default is 'Dec'.

        Returns
        -------
        seasonal_data : xr.dataarray
            array that contains only data within month-range.

        """
        seasonal_data = dataarray.sel(time=self._is_in_month_range(dataarray['time.month'], start_month, end_month))
        return seasonal_data

    def get_mean_loc(self, idx_lst):
        """
        Gets a mean location for a list of indices
        """
        lon_arr=[]
        lat_arr=[]
        for idx in idx_lst:
            map_idx=self.get_map_index(idx)
            lon_arr.append(map_idx['lon'])
            lat_arr.append(map_idx['lat'])
        mean_lat=np.mean(lat_arr)
        
        if max(lon_arr)-min(lon_arr)>180:
            lon_arr=np.array(lon_arr)
            lon_arr[lon_arr<0]=lon_arr[lon_arr<0]+360

        mean_lon=np.mean(lon_arr)
        if mean_lon>180:
            mean_lon-=360
        return(mean_lat, mean_lon)

    def get_locations_in_range(self, lon_range, lat_range, def_map):
        """
        Returns a map with the location within certain range
        """
        
        if (max(lon_range)- min(lon_range) < 180):
            mask = (
                (def_map['lat'] >= min(lat_range) )
                & (def_map['lat'] <= max(lat_range) )
                & (def_map['lon'] >= min(lon_range) )
                & (def_map['lon'] <= max(lon_range) )
                )
        else:   # To account for areas that lay at the border of -180 to 180
            mask = (
                (def_map['lat'] >= min(lat_range) )
                & (def_map['lat'] <= max(lat_range) )
                & ( (def_map['lon'] <= min(lon_range)) | (def_map['lon'] >= max(lon_range)) )
                )
            
        return xr.where(mask, def_map, np.nan)

    def get_idx_range(self, lon_range, lat_range, def_map=None):
        """
        Gets the indices for a specific monsoon from monsoon dictionary keys.
        E.g. can be applied to get all indices of the South American monsoon defined by Wang/EE.
        """
        if def_map is None:
            def_map=self.mask
        mmap=self.get_locations_in_range(lon_range=lon_range,lat_range=lat_range, def_map=def_map)
        ids=np.where(self.flatten_array(mmap, time=False, check=False) == 1)[0]
        
        return ids


    def create_event_time_series(self, th=1, q=0.95, th_eev=15, min_evs=20,  ):

        # Remove days without rain
        data_above_th = self.dataarray.where(self.dataarray>th)
        # Compute percentile data, remove all values below percentile, but with a minimum of threshold q
        print(f"Start remove values below q={q} and at least with q_value >= {th_eev} ...")
        # Gives the quanile value for each cell
        q_mask= data_above_th.quantile(q, dim='time')
        mean_val=data_above_th.mean(dim='time')
        q_median=data_above_th.quantile(0.5, dim='time')
        # Set values below quantile to 0
        data_above_quantile=xr.where(data_above_th>q_mask[:], data_above_th, np.nan)
        # Set values to 0 that have not at least the value th_eev
        data_above_quantile=xr.where(data_above_quantile>th_eev, data_above_quantile, np.nan)
        # Remove cells with less than min_ev events. 
        print(f"Remove cells without min number of events: {min_evs}")
        num_non_nan_occurence=data_above_quantile.count(dim='time')
        # Get relative amount of q rainfall to total yearly rainfall
        rel_frac_q_map= data_above_quantile.sum(dim='time') / self.dataarray.sum(dim='time')
        # Create mask for which cells are left out
        mask=( num_non_nan_occurence > min_evs)
        final_data=data_above_quantile.where(mask, np.nan)
        data_mask=xr.where(num_non_nan_occurence>min_evs, 1, np.nan)


        print("Now create binary event series!")
        event_series=xr.where(final_data[:]>0, 1, 0)
        print("Done!")
        event_series=event_series.rename('evs')
        self.mask=data_mask
        self.q_mask=q_mask
        self.q_median=q_median
        self.mean_val=mean_val
        self.num_eev_map=num_non_nan_occurence
        self.rel_frac_q_map=rel_frac_q_map
        return event_series
    
    def get_ts_of_indices(self,indices ):
        es_time_series=self.flatten_array(dataarray=self.data_evs, check=False ).T[indices]
        
        return es_time_series

    def save(self, filepath):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        if os.path.exists(filepath):
            print("File" + filepath + " already exists! It will be overwritten!")
            os.remove(filepath)
            # os.rename(filepath, filepath + "_backup") # Check why this leads to wrong write below
            
        if (self.anomalies is True) and (self.evs is True):
            dataarray=xr.merge([self.dataarray, self.data_anomalies, self.data_evs])
        elif self.anomalies is True:
            dataarray=xr.merge([self.dataarray, self.data_anomalies])
        elif self.evs is True:
            dataarray=xr.merge([self.dataarray, self.data_evs])
        else:
            dataarray=self.dataarray
        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "anomalies": int(self.anomalies),
            "lsm": int(self.lsm),
            "evs": int(self.evs)
        }
        dataarray.attrs = param_class
        dataarray.to_netcdf(filepath)
        print(f'Stored to File: {filepath}')
        return None

    def same_size_arr(self, arr_arr):
        len_arr=[]
        for arr in arr_arr:
            len_arr.append(len(arr))
        min_len=np.min(len_arr)
        arr_arr_new=[]
        for arr in arr_arr:
            arr_arr_new.append(arr[0:min_len])
        return arr_arr_new

    ##############################################################################
    # Plotting routines from here on
    ##############################################################################


    def make_colorbar(self, ax, cmap, **kwargs):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib as mpl

        divider = make_axes_locatable(ax)
        orientation = kwargs.pop('orientation', 'vertical')
        if orientation == 'vertical':
            loc = 'right'
        elif orientation == 'horizontal':
            loc = 'bottom'

        label= kwargs.pop('label',None) 
        ticks= kwargs.pop('ticks',None) 
        extend=kwargs.pop('extend','neither')
        cax = divider.append_axes(loc, '5%', pad='3%', axes_class=mpl.pyplot.Axes)
        
        return ax.get_figure().colorbar(cmap, cax=cax, orientation=orientation,
                                label=label, shrink=0.8, ticks=ticks,
                                extend=extend,)
    
    def plt_mask_on_map(self, ax, projection):
        left_out=xr.where(np.isnan(self.mask), 1, np.nan)
        ax.contourf(self.dataarray.coords['lon'], self.dataarray.coords['lat'], 
                    left_out,2, hatches=[ '...', '...',], colors='none',extend='lower',
                    transform=projection)

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
        
        print(normticks)
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
            normticks= self.discrete_norm_ticks(vmin, vmax,  num_ticks=num_ticks, shift_ticks=shift_ticks)
            cmap, norm= self.discrete_cmap( vmin,vmax, colormap=color, num_ticks=num_ticks, shift_ticks=shift_ticks)
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
            self.plt_mask_on_map(ax=ax,projection=projection)       
            
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
    