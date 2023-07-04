# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:17:31 2023

@author: konings
"""

import numpy as np
import pandas as pd
from osgeo import gdal
import os
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent, show
import matplotlib.pyplot as plt

def plot_map(arrayToPlot, pwsExtent, stateBorders, title = None, vmin = None, vmax = None, clrmap = 'YlGnBu', savePath = None):
    '''make map with state borders'''
    
    #preliminary calculatios
    statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']    
    
    #actual plotting
    fig, ax = plt.subplots()
    if vmin != None and vmax != None:
        ax = rasterio.plot.show(arrayToPlot, interpolation='nearest', vmin=vmin, vmax=vmax, extent=pwsExtent, ax=ax, cmap=clrmap)
    else:
        ax = rasterio.plot.show(arrayToPlot, interpolation='nearest', extent=pwsExtent, ax=ax, cmap=clrmap)
    stateBorders[stateBorders['NAME'].isin(statesList)].boundary.plot(ax=ax, edgecolor='black', linewidth=0.5) 
    im = ax.get_images()[0]
    #cbar = plt.colorbar(im, ax=ax) #ticks=range(0,6)
    #cbar.ax.set_xticklabels([ 'Deciduous','Evergreen','Mixed','Shrub','Grass', 'Pasture'])
    plt.title(title)
    ax.axis('off')
    plt.xticks([])
    plt.yticks([])
    if savePath != None:
        plt.savefig(savePath)
    plt.show() 
    
    
#prep plotting
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWS_through2021_DecThruMay.tif'
pwsMap =rxr.open_rasterio(pwsPath, masked=True).squeeze()

statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']
ds = gdal.Open(pwsPath)
geotransform = ds.GetGeoTransform()
statesPath = "C:/repos/data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp"
states = gpd.read_file(statesPath)    
#Map = df_to_raster(shapdf['shap_vanGen_n'], np.shape(pwsDecThruMayRaster), lat, lon, geotransform)    

pwsExtent = plotting_extent(pwsMap, pwsMap.rio.transform())


#create basal area arrays
def getRaster(filename):
    ds = gdal.Open(filename)
    data = ds.GetRasterBand(1).ReadAsArray()    
    ds = None
    return data



BALivePath = 'C:/repos/pws_drivers/data/FIABasalAreaAc.tif'
BALiveMap = getRaster(BALivePath)
BAPath = 'C:/repos/pws_drivers/data/FIABasalArea.tif'
BAMap = getRaster(BAPath)
speciesPath = 'C:/repos/pws_drivers/data/FIADomSpecies.tif'
speciesMap = getRaster(speciesPath)

#mask areas outside of PWS states
BALiveMap[np.isnan(pwsMap)] = np.nan
BAMap[np.isnan(pwsMap)] = np.nan
speciesMap[np.isnan(pwsMap)] = np.nan

plot_map(BAMap, pwsExtent, states, 'Basal Area, all species', vmin=0, vmax=250)
plot_map(BALiveMap, pwsExtent, states, 'Basal Area, v2, all species', vmin=0, vmax=250)

#mask areas where dominant speices is not one of the six most commmon
#slow nad not pythonic but whatever
commonSpecList = {65, 69, 122, 202, 756, 64, 106, 108}
cntr = 0
for spec in np.unique(speciesMap):
    if spec not in commonSpecList:
        speciesMap[speciesMap==spec] = np.nan
    else:
        speciesMap[speciesMap==spec] = cntr
        cntr += 1
BAMap[np.isnan(speciesMap)] = np.nan
BALiveMap[np.isnan(speciesMap)] = np.nan

plot_map(speciesMap, pwsExtent, states, clrmap='Dark2', vmin=0, vmax=8)
plot_map(BAMap, pwsExtent, states, 'Basal Area', vmin=0, vmax=350)
plot_map(BALiveMap, pwsExtent, states, 'Basal Area, v2', vmin=0, vmax=350)
print(np.sum(~np.isnan(BAMap)))