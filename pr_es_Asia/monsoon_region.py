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
sys.path.append(PATH + "/../") # Adds higher directory 

from src.dataset import BaseDataset
from src.es_graph_tool import ES_Graph_tool

#%%
"""Class of monsoon regions, containing the
monsoon definitions, the node ids and the regional monsoons."""

class Monsoon_Region(ES_Graph_tool):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str  
        filename  
    var_name: str
        Variable name of interest
    """

    def __init__(self, nc_file, network_file, var_name=None,
                 lon_range=[-180, 180], lat_range=[-90, 90],
                 time_range=['1997-01-01', '2019-01-01'],
                 grid_step=1.0, name='pr', anomalies=False,
                 lsm=False,
                 abs_th_wang=2, abs_th_ee=50, rel_th=0.55, 
                 load=False):
        super().__init__(nc_file, network_file=network_file, var_name=var_name,
                 lon_range=lon_range, lat_range=lat_range,
                 time_range=time_range, grid_step=grid_step,
                 name=name,
                 anomalies=anomalies, lsm=lsm, evs=True,
                 load=load)

        # Define certain monsoon region roughly based on their lon-lat range
        self.boreal_winter=('Dec', 'Mar')    
        self.boreal_summer=('Jun', 'Sep')
        self.monsoon_south_africa=[r'South Africa', (-30,-5),(0,60), 'tab:blue', 'South_Africa' ]
        self.monsoon_north_africa=[r'North Africa', (0,30),(-20,60), 'tab:orange', 'North_America']
        self.monsoon_east_asia=[r'East Asia', (20,35), (112,120), 'tab:green', 'East_Asia' ]
        self.monsoon_india_south_asia=[r'India South Asia', (10,20), (70,120), 'y', 'India_South_Asia' ]
        self.monsoon_india=[r'India', (10,25), (70,85), 'tab:red', 'India' ]
        self.monsoon_south_india=[r'South India', (10,20), (70,85), 'tab:blue', 'South_India' ]
        self.monsoon_north_india=[r'North India', (20,30), (72,85), 'cyan', 'North_India' ]
        self.monsoon_south_asia=[r'South Asia', (10,20), (95,110), 'm', 'South_Asia' ]
        self.monsoon_south_america=[r'South America', (-0,-30), (-80,-40),'magenta', 'South_America' ]
        self.monsoon_nourth_america=[r'North America', (5,30), (-120,-80), 'tab:cyan', 'North_America' ]
        self.monsoon_australia=[r'Australia', (-5,-30), (100,140), 'tab:olive', 'Australia' ]
        self.monsoon_central_west_pacific=[r'CWP', (20,30), (120,130), 'blue', 'Central_West_Pacific' ]
        self.monsoon_central_pacific=[r'CP', (-5,-30), (-120,160), 'green', 'Central_Pacific' ]
        self.itcz_tropics=[r'Tropics', (-5,5), (10,40),'tab:brown','Tropics' ]
        
        self.wang_def,_, _, _=self.monsoon_regions(dtype='pr',abs_th=abs_th_wang, rel_th=rel_th)
        self.ee_def,_,_, _=self.monsoon_regions(dtype='ee',abs_th=abs_th_ee, rel_th=rel_th)
        self.monsoon_dictionary=self.get_monsoon_dict()
    
    def get_sel_mdict(self, hem=None, m_arr=None):
        def removekey(d, key):
            r = dict(d)
            del r[key]
            return r
        m_dict=dict()
        if m_arr is None:
            if hem=='NH':
                m_arr= ['North America', 'North Africa', 'India', 'East Asia']
            elif hem=='SH':
                m_arr= ['South America', 'South Africa', 'Australia']
            elif hem=='Pacific':
                m_arr= ['CP', 'CWP']
            else:
                raise ValueError(f"This hemisphere {hem} does not exist!") 
        for key in m_arr:
            m_dict[key]= self.monsoon_dictionary[key]
        return m_dict
        
    def summer_data(self,data):
        NH_data=self.get_month_range_data(data, start_month=self.boreal_summer[0], end_month=self.boreal_summer[1])
        SH_data=self.get_month_range_data(data, start_month=self.boreal_winter[0], end_month=self.boreal_winter[1])
        return NH_data, SH_data

    def winter_data(self,data):
        NH_data=self.get_month_range_data(data, start_month=self.boreal_winter[0], end_month=self.boreal_winter[1])
        SH_data=self.get_month_range_data(data, start_month=self.boreal_summer[0], end_month=self.boreal_summer[1])
        return NH_data, SH_data

    def NH_SH_data(self, data, season='summer'):
        if season=='summer':
            NH_data, SH_data=self.summer_data(data)
        elif season=='winter':
            NH_data, SH_data=self.winter_data(data)
        else:
            raise ValueError("The season {season} does not exist!")
        return NH_data, SH_data

    def compute_yearly_sum(self, dataarray):
        time_resample="1Y"
        data_sum=dataarray.resample(time=time_resample).sum()
        data_sum_mean=data_sum.mean(dim='time')
        return data_sum_mean

    def rel_fraction(self, data, full_year):

        av_year_sum=self.compute_yearly_sum(full_year)
        av_data_sum=self.compute_yearly_sum(data)
        
        rel_map=av_data_sum / av_year_sum
        
        return rel_map

    def get_diff_rel_maps(self,data, season='summer', dtype='pr'):
        full_year=data
        NH_data, SH_data=self.NH_SH_data(data, season=season)
        
        if dtype=='pr':
            NH_type=NH_data.mean(dim='time')
            SH_type=SH_data.mean(dim='time')
        elif dtype=='ee':
            NH_type=NH_data.where(NH_data>0).count(dim='time')
            SH_type=SH_data.where(SH_data>0).count(dim='time')
        else:
            raise ValueError(f"Data type {dtype} not known!")

        # Get Difference between Summer and Winter 
        diff_map=NH_type - SH_type

        # Get relative difference for 55%
        rel_map_NH=self.rel_fraction(NH_data, full_year)
        rel_map_SH=self.rel_fraction(SH_data, full_year)
   
        rel_map_combined=xr.where(abs(rel_map_NH)>abs(rel_map_SH), abs(rel_map_NH),rel_map_SH )
        
        return diff_map, rel_map_combined, rel_map_NH, rel_map_SH

    def monsoon_regions(self, dtype='pr', abs_th=2, rel_th=0.55):
        if dtype=='pr':
            data=self.dataarray
        elif dtype=='ee':
            data=self.data_evs
        else:
            raise ValueError(f"Data type {dtype} not known!")    

        diff_map, rel_map,_ , _=self.get_diff_rel_maps(data, season='summer', dtype=dtype)

        # Get difference above absolute threshold
        diff_map=xr.where(abs(diff_map)>abs_th, (diff_map),np.nan )
        # Get relative difference for 55%
        rel_map=xr.where(abs(rel_map)>rel_th, abs(rel_map),np.nan )

        # Now combine diff_map and rel_map
        m_def=xr.where((rel_map>0) & (abs(diff_map)>0), 1, np.nan)
        diff_rel_map=xr.where((rel_map>0) & (abs(diff_map)>0), diff_map, 0)
        
        return m_def, diff_rel_map, diff_map, rel_map


    def vis_annomalous_regions(self,data1, data2):
        map0=data1*data2
        map1=(1-data1) * data2
        map2=(1-data2) * data1
        map3=(1-data1) * (1-data2)

        labels=['RF_pr*RF_ee','(1-RF_pr)*RF_ee', '(1-RF_ee)*RF_pr', '(1-RF_pr)*(1-RF_ee)' ]

        return [map0, map1, map2, map3], labels
    
    
    def get_m_locations_in_range(self, monsoon, def_map):
        
        lat_range=monsoon['lat_range']
        lon_range=monsoon['lon_range']
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

    def get_idx_monsoon(self, monsoon, def_map):
        """
        Gets the indices for a specific monsoon from monsoon dictionary keys.
        E.g. can be applied to get all indices of the South American monsoon defined by Wang/EE.
        """
        mmap=self.get_m_locations_in_range(monsoon, def_map)
        ids=np.where(self.flatten_array(mmap, time=False, check=False) == 1)[0]
        return ids

    def get_monsoon_dict(self):
        
        monsoon_dictionary = dict()
        for monsoon in [self.monsoon_north_africa, self.monsoon_south_africa, 
                        self.monsoon_nourth_america, self.monsoon_south_america, 
                        self.monsoon_india_south_asia , self.monsoon_india,self. monsoon_south_asia,
                        self.monsoon_north_india, self.monsoon_south_india,
                        self.monsoon_east_asia, self.monsoon_australia, 
                        self.monsoon_central_pacific, self.monsoon_central_west_pacific]:

            this_monsoon_dictionary=dict()
            for idx, item in enumerate(['name', 'lat_range', 'lon_range', 'color', 'sname']):
                this_monsoon_dictionary[item]=monsoon[idx]

            monsoon_dictionary[monsoon[0]] = this_monsoon_dictionary

        wang_monsoon_regions=self.wang_def
        ee_monsoon_regions=self.ee_def
        for name, monsoon in monsoon_dictionary.copy().items():
            lat_range = monsoon['lat_range']
            lon_range = monsoon['lon_range']

            if (self.check_range(lon_range, self.lon_range) ) and (self.check_range(lat_range, self.lat_range)):
                monsoon['node_ids_wang']=self.get_idx_monsoon(monsoon, wang_monsoon_regions)
                monsoon['node_ids_ee']= self.get_idx_monsoon(monsoon, ee_monsoon_regions)

                # Representative Ids for every location
                m_ids_lst=monsoon['node_ids_ee']
                mean_loc=self.get_mean_loc(m_ids_lst)
                rep_ids=self.get_n_ids(mean_loc)
                monsoon['rep_ids']=np.array(rep_ids)
            else:
                del monsoon_dictionary[name]
            
        if not monsoon_dictionary:
            raise ValueError("ERROR! The monsoon dictionary is empty!")

        return monsoon_dictionary

    def check_range(self, arr_range, compare_arr_range):
        min_arr=min(arr_range)
        max_arr=max(arr_range)

        min_carr=min(compare_arr_range)
        max_carr=max(compare_arr_range)

        if min_arr<min_carr or max_arr > max_carr:
            return False
        else:
            return True 

    def get_m_ids(self, mname, defn='ee'):
        if defn == 'ee':
            m_node_ids=self.monsoon_dictionary[mname]['node_ids_ee']
        elif defn== 'wang':
            m_node_ids=self.monsoon_dictionary[mname]['node_ids_wang']
        else:
            raise ValueError(f"This definition for ids does not exist: {defn}!")
        return m_node_ids

    def get_n_ids(self,loc):
        """
        Gets for a specific location, the neighboring lats and lons ids.
        ----
        Args:
        loc: (float, float) provided as lat, lon values
        """
        slat, slon=loc
        lon=self.grid['lon']
        lat=self.grid['lat']
        sidx=self.get_index_for_coord(lon=slon, lat=slat)
        idx_lon,_=self.find_nearest(lon, slon)
        idx_lat,_=self.find_nearest(lat, slat)
        
        n_locs=[(lon[idx_lon], lat[idx_lat+1] ), 
                (lon[idx_lon], lat[idx_lat-1] ),
                (lon[idx_lon+1], lat[idx_lat] ),
                (lon[idx_lon-1],lat[idx_lat] )
                ]
        n_idx=[sidx]
        for nloc in n_locs:
            n_idx.append(self.get_index_for_coord(lon=nloc[0], lat=nloc[1]))
        
        return n_idx
    



# %%    
if __name__ == "__main__":
    cluster = True
    grid_step = 2.5
    vname = 'pr'
    num_cpus = 64
    num_jobs=16
    
    if os.getenv("HOME") =='/home/goswami/fstrnad80':
        dirname = "/home/goswami/fstrnad80/data/GPCP/"
    else:
        dirname = "/home/strnad/data/GPCP/"
        # dirname = "/home/jakob/climate_data/local/"

    fname = dirname +"gpcp_daily_1996_2020_2p5_new.nc4"
    print('Loading Data')
    ds = Monsoon_Region(fname, var_name=vname, 
                         time_range=['1997-01-01', '2019-01-01'],
                         grid_step=grid_step)
    monsoon_dict=ds.monsoon_dictionary
    ee_diff_map , ee_rel_map, ee_summer_rel_map, ee_winter_rel_map=ds.get_diff_rel_maps(ds.data_evs, season='summer', 
                                                                                        dtype='ee')
    pr_diff_map , pr_rel_map, pr_summer_rel_map, pr_winter_rel_map=ds.get_diff_rel_maps(ds.dataarray, season='summer', 
                                                                        dtype='pr')
    
    # %%
    # Extrem Event Definition
    ee_def, diff_rel_map, diff_map, rel_map =ds.monsoon_regions(dtype='ee', rel_th=0.55, abs_th=60)
    title="Extrem Event definition"
    ds.plot_contour_map(ee_def, color='Blues',bar=False, title=title)
    # %%
    # Rel Map EE
    title="Relative Fraction Summer (JJA) / Full Year Extrem Events "

    ds.plot_contour_map(rel_map, color='coolwarm', title=title, n_contour=11, vmin=0, vmax=1, 
                        bar=True, clabel=False, 
                        label='Relative number of Extrem Events')
    plt.savefig(PATH + f"/../plots/def_eev_grid_{grid_step}_ES.pdf")
    # %%
    # Diff Map
    title="Difference Summer (JJA) - Winter (DJF) Extrem Events "
    ds.plot_contour_map(ee_diff_map, color='coolwarm', title=title, n_contour=11, vmin=-150, vmax=150, 
                        bar=True, clabel=False, 
                        label='Difference EE JJA - DJF')
    plt.savefig(PATH + f"/../plots/diff_eev_grid_{grid_step}_ES.pdf")

    # %% 
    # Wang Monsoon Regions
    wang_def, diff_rel_map, diff_map, rel_map =ds.monsoon_regions(dtype='pr', rel_th=0.55, abs_th=2)
    title="Bin Wang definition"

    ds.plot_contour_map(wang_def, color='Blues',bar=False, title=title)

    plt.savefig(PATH + f"/../plots/def_wang_grid_{grid_step}_ES.pdf")
    # %%
    # Diff Map
    
    title="Difference Summer (JJA) - Winter (DJF) in precipitation "
    ds.plot_contour_map(pr_diff_map, color='coolwarm', title=title, n_contour=11, vmin=-10, vmax=10, 
                        bar=True, clabel=False, 
                        label='Difference pr JJA - DJF [mm/day]')
    plt.savefig(PATH + f"/../plots/wang_diff_grid_{grid_step}_ES.pdf")

    # %% 
    # Rel Map
    title="Relative fraction of Summer rainfalls on annual precipitation "    
    ds.plot_contour_map(pr_rel_map, color='coolwarm', title=title, n_contour=11, vmin=0, vmax=1, 
                        bar=True, clabel=False, 
                        label='relative fraction')
    plt.savefig(PATH + f"/../plots/wang_relative_fraction_grid_{grid_step}_ES.pdf")


    # Diff Rel Map Summer/Winter Wang

    title="Relative Fraction Summer (JJA) / Full Year Extrem Events "

    ds.plot_contour_map(pr_summer_rel_map, color='coolwarm', title=title, n_contour=11, vmin=0, vmax=1, 
                        bar=True, clabel=False, 
                        label='Relative number of Extrem Events')
    plt.savefig(PATH + f"/../plots/summer_rel_frac_pr_grid_{grid_step}_ES.pdf")

    title="Relative Fraction Winter (DJF) / Full Year Extrem Events "

    ds.plot_contour_map(pr_winter_rel_map, color='coolwarm', title=title, n_contour=11, vmin=0, vmax=1, 
                        bar=True, clabel=False, 
                        label='Relative number of Extrem Events')
    plt.savefig(PATH + f"/../plots/winter_rel_frac_pr_grid_{grid_step}_ES.pdf")

    # %%
    import cartopy.crs as ccrs
    nrows=2
    ncols=2
    projection=ccrs.Mollweide(central_longitude=0)
    maps=[pr_summer_rel_map, pr_winter_rel_map, ee_summer_rel_map, ee_winter_rel_map]
    labels=['RF Pr Summer (JJA)', 'RF Pr Winter (DJF)', 'RF EE Summer (JJA)', 'RF EE Winter (DJF)']

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,6*nrows),
                           subplot_kw=dict(projection=projection ) )
    
    label='Relative Fraction of season / full year'
    for idx,thisMap in enumerate(maps):
        i = int(idx/ncols)
        j= idx-ncols*i
        print(i,j)
        this_ax=ax[i][j]
        this_label=labels[idx]
        _, cbar=ds.plot_contour_map(thisMap, color='coolwarm', ax=this_ax,fig=None,
                        title=this_label, n_contour=11, vmin=0, vmax=None, 
                        bar=False, clabel=False, 
                        label=None)
    fig.colorbar(cbar, ax=ax.ravel().tolist(), extend='both', orientation='horizontal',
                            label=label, shrink=0.7)
    
    plt.savefig(PATH + f"/../plots/rel_frac_summer_winter_grid_{grid_step}_ES.pdf")


    # %% 
    # Anomalous Regions
    nrows=2
    ncols=2
    projection=ccrs.Mollweide(central_longitude=0)
    an_maps, labels=ds.vis_annomalous_regions(pr_summer_rel_map, ee_summer_rel_map)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,6*nrows),
                           subplot_kw=dict(projection=projection ) )
    
    label='Relative Fraction of season / full year'
    for idx,thisMap in enumerate(an_maps):
        i = int(idx/ncols)
        j= idx-ncols*i
        print(i,j)
        this_ax=ax[i][j]
        this_label=labels[idx]
        _, cbar=ds.plot_contour_map(thisMap, color='coolwarm', ax=this_ax,fig=fig,
                        title=this_label, n_contour=11, vmin=0, vmax=None, 
                        bar=True, clabel=False, 
                        label=None)
    # fig.colorbar(cbar, ax=ax.ravel().tolist(), extend='both', orientation='horizontal',
    #                         label=label, shrink=0.7)
    fig.suptitle("Summer (JJA)")
    plt.savefig(PATH + f"/../plots/anomalous_summer_rel_frac_grid_{grid_step}_ES.pdf")






    # %%
    m_name='North America'
    mregion=monsoon_dict[m_name]
    link_map = ds.get_map(ds.flat_idx_array(mregion['node_ids_wang']))
    ds.plot_map(link_map, color='Reds', title=m_name)
    plt.savefig(PATH + f"/../plots/{vname}_grid_{grid_step}_{m_name}.pdf")

    # %%
    # create network
    Precip_ESNet = ds.create_es_network(ds, vname, grid_step, linkbund=True, savenet=True, num_jobs=num_jobs)

    # %% 
    # Plot Q values 
    
    q_map = ds.q_mask
    ds.plot_map(q_map, color='Blues', label=f'Quantile {ds.q} values [mm/days]')
    plt.savefig(PATH + f"/../plots/{vname}_grid_{grid_step}_q_mask.pdf")
    # %% 
    # Plot  mask
    mask=ds.mask
    ds.plot_map(mask, color='Blues', label=f'Definition values')
    plt.savefig(PATH + f"/../plots/{vname}_grid_{grid_step}_ES_mask.pdf")





