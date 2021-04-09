#import pygrib

from netCDF4 import Dataset

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xarray as xr
import xarray.ufuncs as xu
import matplotlib.pyplot as plt

import matplotlib.colors as colors

import numpy as np

#This function will load your data from location specified in fn as an xrarray dataset
def load():
    fn = '/home/lea/Dokumente/Uni/Masterthesis/Data/ERA5/Ausprobierdaten/daily_surface_u_v_2019.nc'
    ds = xr.load_dataset(fn)
    return ds

#This function returns a dataset for a specified latitude-range given a global dataset
def getRangeLat(data, latlim1, latlim2):
    #Get data for latlim1 to latlim2 in latitudes return as xarray dataset
    dataLat = data.sel(latitude=slice(latlim2,latlim1))
    return dataLat

#This function returns a dataset for a specified time given the dataset
def getRangeTime(data, time):
    dataMonth = data.sel(time=slice(time))
    return dataMonth

#This function returns a dataset for a specified longitude-range given a global dataset
def getRangeLong(data, longlim1, longlim2):
    dataLong = data.sel(longitude=slice(longlim1,longlim2))
    return dataLong

#draw Wind arrows and windspeed
def drawData(data):
    lats= data['latitude']
    longs= data['longitude']
    u= data['u10']
    v= data['v10']
    print(u)

    # Create a cube containing the wind speed
    windspeed = (u ** 2 + v ** 2) ** 0.5
    windspeed.rename('windspeed')

    fig = plt.figure()
    ax1 = plt.axes(projection=ccrs.PlateCarree())

    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)

    # map extent, [long,long,lat,lat]
    ax1.set_extent([-30, 0, -10, 10], ccrs.PlateCarree())

    # Plot the wind speed as a contour plot
    plt.contourf(longs, lats, windspeed[0,:,:], transform=ccrs.PlateCarree())

    # Normalise the data for uniform arrow size
    u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
    v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)

    latsteps= 1 #1 latitude = 4 steps, ! single values for lat, no mean
    longsteps = 1

    #change scale for arrow length (bigger scale -> shorter arrow), width for line thickness and headwidth as factor of width
    ax1.quiver(longs[::longsteps], lats[::latsteps], u_norm[0,::latsteps, ::longsteps], v_norm[0, ::latsteps, ::longsteps],pivot='middle',
                                            transform=ccrs.PlateCarree(), scale=70, width=0.001, headwidth=2.5)
    plt.colorbar()

    plt.show()
    return

data = load()
#latsmonth= getRangeLat(data, -10,10)
#longsmonth=getRangeLong(data, 10, 40)
latslongsmonth = getRangeTime(data, "2019-01-01")

drawData(latslongsmonth)