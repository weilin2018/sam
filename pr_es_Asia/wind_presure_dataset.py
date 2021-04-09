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

from src.dataset import BaseDataset


class Wind_dataset(BaseDataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str  
        filename  
    var_name: str
        Variable name of interest
    """

    def __init__(self, nc_file_u, nc_file_v, var_name='windspeed',
                 lon_range=[-180, 180], lat_range=[-90, 90],
                 time_range=['1998-01-01', '2019-01-01'],
                 grid_step=1.0, anomalies=False, evs=False, name=None, lsm=False, rrevs=False, load=False):
            
        self.ds_uwind = BaseDataset(nc_file_u, var_name='u', name=name,
                        lon_range=lon_range, lat_range=lat_range,
                        time_range=time_range,
                        grid_step=grid_step,
                        lsm=lsm, 
                        evs=evs,
                        rrevs=rrevs,
                        anomalies=anomalies,
                        load=load)
        self.ds_vwind = BaseDataset(nc_file_v, var_name='v', name=name,
                        lon_range=lon_range, lat_range=lat_range,
                        time_range=time_range,
                        grid_step=grid_step,
                        lsm=lsm, 
                        evs=evs, 
                        rrevs=rrevs,
                        anomalies=anomalies,
                        load=load)
    
        self.u=self.ds_uwind.dataarray 
        self.v=self.ds_vwind.dataarray 
        windspeed= (self.u ** 2 + self.v ** 2) ** 0.5
        self.windspeed=windspeed.rename('windspeed')

        self.windspeed_anomalies=self.compute_anomalies(self.windspeed, group='dayofyear')

    def save(self, filepath, filepath_u, filepath_v):
        """Save the dataset class of wind u,v and windspeed to file.
        Args:
        ----
        filepath: str
        """
        self.ds_uwind.save(filepath_u)
        self.ds_vwind.save(filepath_v)

        if os.path.exists(filepath):
            print("File" + filepath + " already exists! It will be overwritten!")
            os.remove(filepath)
            # os.rename(filepath, filepath + "_backup") # Check why this leads to wrong write below
        
        if self.ds_uwind.anomalies is True and self.ds_vwind.anomalies is True:
            dataarray=xr.merge([self.windspeed, self.windspeed_anomalies])
        dataarray.to_netcdf(filepath)
        print(f'Stored to File: {filepath}')
        return None


# %%    
if __name__ == "__main__":
    
    
    load_ds=False
    
    grid_step = 1
    vname_psurf = 'sp'
    anomalies=True
    plevel=250

    wind_name='wind_era5'
    num_cpus = 64
    time_range=['1998-01-01', '2019-12-31']
    
    # Asia subcontinent range
    lat_range=[-15, 45]
    lon_range=[55, 150]
    if os.getenv("HOME") =='/home/goswami/fstrnad80':
        dirname_wind = "/mnt/qb/goswami/fstrnad80/era5/wind/850_hpa/"
        dirname_uwind = f"/mnt/qb/goswami/fstrnad80/era5/u_component_of_wind/{plevel}/"
        dirname_vwind = f"/mnt/qb/goswami/fstrnad80/era5/v_component_of_wind/{plevel}/"

    else:
        dirname_wind = "/home/strnad/data/wind/850_hpa/"
        dirname_uwind = f"/home/strnad/data/era5/u_component_of_wind/{plevel}/"
        dirname_vwind = f"/home/strnad/data/era5/v_component_of_wind/{plevel}/"
        num_cpus = 16

    # fname_u = dirname_wind +f'u_daily_1980_2020.nc'
    # fname_v = dirname_wind +f'v_daily_1980_2020.nc'
    fname_u = dirname_uwind +f'u_component_of_wind_2019_{plevel}_1990_2020.nc'
    fname_v = dirname_vwind +f'v_component_of_wind_2019_{plevel}_1990_2020.nc'
    
    # %%
    print('Loading Data')
    dataset_file_u = PATH + f"/../../outputs/{wind_name}_u_{plevel}_ds_{grid_step}.nc"
    dataset_file_v = PATH + f"/../../outputs/{wind_name}_v_{plevel}_ds_{grid_step}.nc"
    dataset_file_ws = PATH + f"/../../outputs/{wind_name}_ws_{plevel}_ds_{grid_step}.nc"


    if anomalies is True:
        dataset_file_u = PATH + f"/../../outputs/{wind_name}_u_{plevel}_ds_{grid_step}_anomalies.nc"
        dataset_file_v = PATH + f"/../../outputs/{wind_name}_v_{plevel}_ds_{grid_step}_anomalies.nc"
        dataset_file_ws = PATH + f"/../../outputs/{wind_name}_ws_{plevel}_ds_{grid_step}_anomalies.nc"


    if load_ds is True:
        # ds_uwind = BaseDataset(dataset_file_u, load=load_ds, rrevs=False, name=wind_name)
        # ds_vwind = BaseDataset(dataset_file_v, load=load_ds, rrevs=False, name=wind_name)

        ds_wind=Wind_dataset(nc_file_u=dataset_file_u, nc_file_v=dataset_file_v, 
                            load=load_ds,name=wind_name)

    else:
        print('Create Wind Dataset')
        ds_wind= Wind_dataset(nc_file_u=fname_u,  nc_file_v=fname_v,  var_name='wind', name=wind_name,
                        lon_range=lon_range, lat_range=lat_range,
                        time_range=time_range,
                        grid_step=grid_step,
                        lsm=False, 
                        evs=False, 
                        anomalies=anomalies)

        ds_wind.save(filepath=dataset_file_ws, filepath_u=dataset_file_u, filepath_v=dataset_file_v)
        
        # ds_vwind = BaseDataset(fname_v, var_name='v', name=wind_name,
        #                 lon_range=lon_range, lat_range=lat_range,
        #                 time_range=time_range,
        #                 grid_step=grid_step,
        #                 lsm=False, 
        #                 evs=False, 
        #                 anomalies=anomalies)
        # ds_vwind.save(dataset_file_v)
        # ds_uwind = BaseDataset(fname_u, var_name='u', name=wind_name,
        #                 lon_range=lon_range, lat_range=lat_range,
        #                 time_range=time_range,
        #                 grid_step=grid_step,
        #                 lsm=False, 
        #                 evs=False,
        #                 anomalies=anomalies)
        # ds_uwind.save(dataset_file_u)

        sys.exit(0)


    # %%
    u=ds_wind.u 
    v=ds_wind.v 
    windspeed=ds_wind.windspeed

    windspeed_anomalies=ds_wind.windspeed_anomalies

    # %%
    # Plot wind
    um=u.mean(dim='time').data 
    vm=v.mean(dim='time').data
    windspeed_anomalies_m=windspeed_anomalies.mean(dim='time')
    # dmap= u / np.sqrt(u ** 2.0 + u ** 2.0)

    longs=u.coords['lon']
    lats=u.coords['lat']
    latsteps=lonsteps=4
    
    u_dat= um[::latsteps, ::lonsteps]
    v_dat= vm[::latsteps, ::lonsteps]

    ax=ds_wind.ds_uwind.plot_map(dmap=windspeed_anomalies_m, color='coolwarm_r', label=f'Mean Wind Speed [m/s]', 
                    projection='PlateCarree', plt_mask=False)

    ax.quiver(longs[::lonsteps], lats[::latsteps], u=u_dat, v=v_dat,
                                 pivot='middle', transform=ccrs.PlateCarree(),
                                 scale=120, width=.005, headwidth=2.5)
    plt.savefig(PATH + f"/../../plots/asia/{wind_name}_grid_{grid_step}_mean.png")

   
    


# %%


    