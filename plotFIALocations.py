# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:08:51 2022

@author: konings
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from osgeo import gdal
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent, show
import pickle

def plot_map(arrayToPlot, pwsExtent, stateBorders, axMap = None, vmin = None, vmax = None, clrmap = 'YlGnBu', savePath = None, legCode = None):
    '''make map with state borders'''
    
    #preliminary calculatios
    statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']    
    
    #actual plotting
    if axMap == None:
        fig, axMap = plt.subplots()
    if vmin != None and vmax != None:
        rasterio.plot.show(arrayToPlot, interpolation='none', vmin=vmin, vmax=vmax, extent=pwsExtent, ax=axMap, cmap=clrmap)
    else:
        rasterio.plot.show(arrayToPlot, interpolation='none', extent=pwsExtent, ax=axMap, cmap=clrmap)
    stateBorders[stateBorders['NAME'].isin(statesList)].boundary.plot(ax=axMap, edgecolor='black', linewidth=0.5) 
    axMap.axis('off')
    plt.xticks([])
    plt.yticks([])
    if legCode == 1:
        black_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=8, label='Mixed species')
        red_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=8, label='Dominant species')
        plt.legend(handles=[black_circle, red_circle], loc='lower left')

    if savePath != None:
        ax.set_rasterized(True)
        plt.savefig(savePath, dpi=300, bbox_inches = "tight")
    plt.show() 
    
    

flpth = 'G:/My Drive/0000WorkComputer/dataStanford/fiaAnna/'

#read two csv files
varsdf = pd.read_csv(flpth + 'CONDVars_LL.csv', sep=',', header=0, 
                     usecols=['CN','INVYR', 'PLOT', 'MEASYEAR', 'CONDID', \
					 'LAT', 'LON', 'FORTYPCD', 'FLDTYPCD', 'CONDPROP_UNADJ','BALIVE'])       
spdf = pd.read_csv(flpth + 'COND_Spp_Live_Ac.csv', sep=',', header=0, 
                     usecols=['COND_CN','SPCD','BALiveAc','TPALiveAc'])
p50df = pd.read_csv(flpth + 'FIA_traits_Alex.csv', sep=',', header=0)

#varsdf has 1,084,063 unique CN values, out of the same number of rows
#spdf has 413,987 unique COND_CN values, out of 1,926,679

#merge dataset to add lat lon information to condition information
combdf = spdf.merge(varsdf, left_on=['COND_CN'], right_on=['CN'], how='inner')

#Load PWS lats and lons, geotransform to figure out how to aggregate
ds = gdal.Open('C:/repos/data/pws_features/PWS_through2021_allSeas.tif')
gt = ds.GetGeoTransform()
pws = ds.GetRasterBand(1).ReadAsArray()
pws_y,pws_x = pws.shape
wkt_projection = ds.GetProjection()
ds = None
                
#read FIA points that are domiant from rf_regression.py
pickleLoc = '../data/df_wSpec.pkl'
rfLocs = pd.read_pickle(pickleLoc)
rfLocs = rfLocs.rename(columns={"lat": "LAT", "lon": "LON"}, errors="raise")
                
#save each dataframe of locations to PWS grid
def makeGridWithPlots(df, gt, pws):
    '''
    Parameters
    ----------
    df : dataframe, assumed to have 'LON' and 'LAT' columns
    gt : geotransform associated with grid
    pws : basis of grid. NaN locatins are assumed to transfer to outgrid

    Returns
    -------
    outGrid
    '''
    indX = np.floor( (df['LON']-gt[0])/gt[1] ).to_numpy().astype(int)
    indY = np.floor( (df['LAT']-gt[3])/gt[5] ).to_numpy().astype(int)
    n1, n2 = pws.shape
    #this has pixels in American Samoa and the like. Remove those
    mask = np.ones(indX.shape, dtype=bool)
    mask[indX < 0] = False
    mask[indX >= n2] = False
    mask[indY < 0] = False
    mask[indY > n1] = False
    indX = indX[mask]
    indY = indY[mask]
    dfMasked = df.copy()[mask]
    #create array, and assign 
    outGrid = np.zeros(pws.shape) * np.nan
    outGrid[indY, indX] = 1
    outGrid[np.isnan(pws)] = np.nan
    return outGrid

domSpecMap = makeGridWithPlots(rfLocs, gt, pws)
FIAPlotMap = makeGridWithPlots(combdf, gt, pws)
#add arrays so you can try to show in one image with different colors
domSpecMap[np.where(np.isnan(domSpecMap))] = 0
FIAPlotMap[np.where(np.isnan(FIAPlotMap))] = 0
combinedMap = domSpecMap + FIAPlotMap
combinedMap[np.where(combinedMap==0)] = np.nan

#plot
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_allSeas_nonorm_4monthslag_exact6years.tif'
pwsMap =rxr.open_rasterio(pwsPath, masked=True).squeeze()
pwsExtent = plotting_extent(pwsMap, pwsMap.rio.transform())
statesPath = "C:/repos/data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp"
states = gpd.read_file(statesPath)  

fig, (ax1, ax2) = plt.subplots(2,1)
plot_map(domSpecMap, pwsExtent, states, clrmap='Dark2', vmin=0, vmax=1)
plot_map(FIAPlotMap, pwsExtent, states, clrmap='Dark2', vmin=0, vmax=1)

#plot in a combined figure where red is locations with a dominant species
#summedMap value of 2, and black is pixels where there is an FIA plot locatin
#but it doesn't have a dominant species (assumedMap value of 1). Colormap 
# and vmin/vmax are chosen so that colors work out to black and red
fig, ax = plt.subplots()
plot_map(combinedMap, pwsExtent, states, clrmap='hot_r', vmin=-1, vmax=2, 
         legCode=1, savePath='C:/repos/figures/PWSDriversPaper/plotMaps.jpeg')
fig, ax = plt.subplots()
plot_map(combinedMap, pwsExtent, states, clrmap='hot_r', vmin=-1, vmax=2, 
         legCode=1, savePath='C:/repos/figures/PWSDriversPaper/plotMaps.pdf')
