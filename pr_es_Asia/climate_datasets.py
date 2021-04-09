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

# %%    
if __name__ == "__main__":
    
    grid_step = 1
    vname_psurf = 'sp'
    vname_sh = 'q'
    vname_gp= 'z'
    anomalies=True
    
    plevels=[250, 850]
    name_psurf='psurf_era5'   
    name_gp='gp_era5'
    name_sh='sh_era5'
    num_cpus = 64
    time_range=['1998-01-01', '2019-12-31']
    
    # Asia subcontinent range
    lat_range=[-15, 45]
    lon_range=[55, 150]

    for plevel in plevels:
        if os.getenv("HOME") =='/home/goswami/fstrnad80':
            dirname_psurf = "/mnt/qb/goswami/fstrnad80/era5/PSurf/"
            dirname_geopot= f"/mnt/qb/goswami/fstrnad80/era5/geopotential/{plevel}/"
            dirname_sh= f"/mnt/qb/goswami/fstrnad80/era5/specific_humidity/{plevel}/"
        else:
            dirname_psurf = "/home/strnad/data/PSurf/"
            dirname_geopot= f"/home/strnad/data/geopotential/{plevel}/"
            dirname_sh= f"/home/strnad/data/specific_humidity/{plevel}/"

            num_cpus = 16

        fname_psurf = dirname_psurf +f'surface_pressure_daily_1980_2020.nc'
        fname_gp= dirname_geopot + f'geopotential_{plevel}_1990_2020.nc'
        fname_sh= dirname_sh + f'specific_humidity_{plevel}_1990_2020.nc'
        

        # %%
        print('Loading Data')
        dataset_file_psurf = PATH + f"/../../outputs/{name_psurf}_{vname_psurf}_ds_{grid_step}.nc"
        dataset_file_gp = PATH + f"/../../outputs/{name_gp}_{plevel}_{vname_gp}_ds_{grid_step}.nc"
        dataset_file_sh = PATH + f"/../../outputs/{name_sh}_{plevel}_{vname_sh}_ds_{grid_step}.nc"

        if anomalies is True:
            dataset_file_psurf = PATH + f"/../../outputs/{name_psurf}_{vname_psurf}_ds_{grid_step}_anomalies.nc"
            dataset_file_gp = PATH + f"/../../outputs/{name_gp}_{plevel}_{vname_gp}_ds_{grid_step}_anomalies.nc"
            dataset_file_sh = PATH + f"/../../outputs/{name_sh}_{plevel}_{vname_psurf}_ds_{grid_step}_anomalies.nc"



        print('Create Dataset')
        
        # ds_psurf= BaseDataset(fname_psurf, var_name=vname_psurf, name=name_psurf,
        #                 lon_range=lon_range, lat_range=lat_range,
        #                 time_range=time_range,
        #                 grid_step=grid_step,
        #                 lsm=False, 
        #                 evs=False, 
        #                 anomalies=anomalies)
        # ds_psurf.save(dataset_file_psurf)

        ds_gp= BaseDataset(fname_gp, var_name=vname_gp, name=name_gp,
                        lon_range=lon_range, lat_range=lat_range,
                        time_range=time_range,
                        grid_step=grid_step,
                        lsm=False, 
                        evs=False, 
                        anomalies=anomalies)
        ds_gp.save(dataset_file_gp)

        ds_sh= BaseDataset(fname_sh, var_name=vname_sh, name=name_sh,
                        lon_range=lon_range, lat_range=lat_range,
                        time_range=time_range,
                        grid_step=grid_step,
                        lsm=False, 
                        evs=False, 
                        anomalies=anomalies)
        ds_sh.save(dataset_file_sh)


