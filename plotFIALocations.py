# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:08:51 2022

@author: konings
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent, show
import pickle

def plot_map(arrayToPlot, pwsExtent, stateBorders, title = None, axMap = None, vmin = None, vmax = None, clrmap = 'YlGnBu', savePath = None):
    '''make map with state borders'''
    
    #preliminary calculatios
    statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']    
    
    #actual plotting
    if axMap == None:
        fig, axMap = plt.subplots()
    if vmin != None and vmax != None:
        rasterio.plot.show(arrayToPlot, interpolation='nearest', vmin=vmin, vmax=vmax, extent=pwsExtent, ax=axMap, cmap=clrmap)
    else:
        rasterio.plot.show(arrayToPlot, interpolation='nearest', extent=pwsExtent, ax=axMap, cmap=clrmap)
    stateBorders[stateBorders['NAME'].isin(statesList)].boundary.plot(ax=axMap, edgecolor='black', linewidth=0.5) 
    #im = axMap.get_images()[0]
    #cbar = plt.colorbar(im, ax=ax) #ticks=range(0,6)
    #cbar.ax.set_xticklabels([ 'Deciduous','Evergreen','Mixed','Shrub','Grass', 'Pasture'])
    #plt.title(title)
    axMap.axis('off')
    plt.xticks([])
    plt.yticks([])
    if savePath != None:
        plt.savefig(savePath)
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
                
#store as pickle file for use elsewhere in mapping/exploring the data sources
pickleLoc = '../data/dominantLocs.pkl'
dominantLocs = pd.read_pickle(pickleLoc)
                
#save each dataframe of locations to PWS grid
def makeGridWithPlots(df, gridCol, gt, pws):
    '''
    Parameters
    ----------
    df : dataframe, assumed to have 'LON' and 'LAT' columns
    gridCol: the column of the dataframe to be gridded
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
    outGrid[indY, indX] = dfMasked[gridCol]   
    outGrid[np.isnan(pws)] = np.nan
    return outGrid

domSpecMap = makeGridWithPlots(dominantLocs, 'SPCD', gt, pws)
FIAPlotMap = makeGridWithPlots(combdf, 'SPCD', gt, pws)

#plot
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWS_through2021_allSeas.tif'
pwsMap =rxr.open_rasterio(pwsPath, masked=True).squeeze()
pwsExtent = plotting_extent(pwsMap, pwsMap.rio.transform())
statesPath = "C:/repos/data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp"
states = gpd.read_file(statesPath)  

fig, (ax1, ax2) = plt.subplots(2,1)
plot_map(domSpecMap, pwsExtent, states, clrmap='Dark2', vmin=0, vmax=8)
plot_map(FIAPlotMap, pwsExtent, states, clrmap='Dark2', vmin=0, vmax=8)

#still need to get to plot as two subplots. Give up for now

