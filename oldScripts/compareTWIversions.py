# -*- coding: utf-8 -*-
"""
Compare old and new TWI files
Created on Thu Jun  8 16:18:43 2023

@author: konings
"""
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent, show

fileDir = 'G:/My Drive/0000WorkComputer/dataStanford/PWSFeatures/resampled/'

#load files
band = 1
dsOld = gdal.Open(fileDir + 'twi.tif')
oldTWI = dsOld.GetRasterBand(band).ReadAsArray()    
dsOld = None

dsNew = gdal.Open(fileDir + 'twi_epsg4326_4000m_merithydro.tif')
newTWI = dsNew.GetRasterBand(band).ReadAsArray() 
geotransform = dsNew.GetGeoTransform()   
dsNew = None

dsSlope = gdal.Open(fileDir + 'usa_slope_project.tif')
slope = dsSlope.GetRasterBand(band).ReadAsArray() 
dsSlope = None

dsVPD = gdal.Open(fileDir + 'vpd_mean.tif')
VPD = dsVPD.GetRasterBand(band).ReadAsArray() 
dsVPD = None

dsNDVI = gdal.Open(fileDir + 'ndvi_mean.tif')
NDVI = dsNDVI.GetRasterBand(band).ReadAsArray() 
dsNDVI = None


dsPWS = gdal.Open('G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_allSeas_nonorm_4monthslag_exact6years.tif')
pws = dsPWS.GetRasterBand(band).ReadAsArray() 
dsPWS = None


#replace with nans
oldTWI[oldTWI==oldTWI[620,0]] = np.nan
newTWI[newTWI==newTWI[620,0]] = np.nan
slope[slope==slope[620,0]] = np.nan
VPD[VPD==VPD[620,0]] = np.nan
NDVI[NDVI==NDVI[620,0]] = np.nan
pws[pws==pws[620,0]] = np.nan

realIndices = np.where( ~np.isnan(oldTWI) & ~np.isnan(newTWI) )
print('R2 = ' + str( np.corrcoef( newTWI[realIndices], oldTWI[realIndices] ) ) )
print('R2 = ' + str(np.corrcoef(newTWI.flatten(),oldTWI.flatten())))

def plot_map(arrayToPlot, pwsExtent, stateBorders, title, vmin = None, vmax = None, clrmap = 'YlGnBu', savePath = None):
    '''make map with state borders'''
    
    #preliminary calculatios
    statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']    
    
    #actual plotting
    fig, ax = plt.subplots()
    if vmin != None and vmax != None:
        ax = rasterio.plot.show(arrayToPlot, vmin=vmin, vmax=vmax, extent=pwsExtent, ax=ax, cmap=clrmap)
    else:
        ax = rasterio.plot.show(arrayToPlot, extent=pwsExtent, ax=ax, cmap=clrmap)
    stateBorders[stateBorders['NAME'].isin(statesList)].boundary.plot(ax=ax, edgecolor='black', linewidth=0.5) 
    im = ax.get_images()[0]
    cbar = plt.colorbar(im, ax=ax) #ticks=range(0,6)
    #cbar.ax.set_xticklabels([ 'Deciduous','Evergreen','Mixed','Shrub','Grass', 'Pasture'])
    plt.title(title)
    ax.axis('off')
    plt.xticks([])
    plt.yticks([])
    if savePath != None:
        plt.savefig(savePath)
    plt.show() 
    
#prep plotting
statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']
#ds = gdal.Open(decThruMayPath)
#geotransform = ds.GetGeoTransform()
statesPath = "C:/repos/data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp"
states = gpd.read_file(statesPath)    
decThruMayPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_DecThruMay.tif'
pwsDecThruMayMap =rxr.open_rasterio(decThruMayPath, masked=True).squeeze()
#pwsExtent = plotting_extent(oldTWI, oldTWI.rio.transform())    
pwsExtent = plotting_extent(pwsDecThruMayMap, pwsDecThruMayMap.rio.transform())    

plot_map(oldTWI, pwsExtent, states, 'old TWI', vmin = 0, vmax = '1500' )
plot_map(newTWI, pwsExtent, states, 'new TWI', vmin = 0, vmax = '150' )
plot_map(slope, pwsExtent, states, 'slope', vmin = 0, vmax = 1 )
plot_map(VPD, pwsExtent, states, 'VPD_mean', vmin = 0, vmax = 30 )
plot_map(NDVI, pwsExtent, states, 'mean NDVI', vmin = 0.2, vmax = 1 )
plot_map(pws, pwsExtent, states, 'PWS', vmin = 0, vmax = 3 )