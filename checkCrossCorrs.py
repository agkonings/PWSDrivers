# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:39:09 2022

attempt to see if there's a cross correlation between VPD-based features and 
DFMC extremes that could somehow mess up the PWS importance analysis

@author: konings
"""
import pandas as pd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_values(filename, mx, my, band = 1):
    """
    Vector implementation of query of raster value at lats and lons

    Parameters
    ----------
    filename : raster path
    mx : Lon values (list or array)
    my : lat values (list or array)
    band : band position to query. int, optional. The default is 1.

    Returns
    -------
    1D array of value of raster at lats and lons

    """
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(band).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int) #x pixel
    py = ((my - gt[3]) / gt[5]).astype(int) #y pixel
    ds = None
    return data[py,px]

def get_lats_lons(data, gt):
    """
    Fetch list of lats and lons corresponding to geotransform and 2D numpy array

    Parameters
    ----------
    data : 2D numpy array. This is the array whose shape will be used to
            generate the list of lats and lons
    gt : 6 size tuple of gdal geotransform

    Returns
    -------
    lats : array of latitudes
    lons : array of longitudes

    """
    x = range(data.shape[1])
    y = range(data.shape[0])
    
    x,y = np.meshgrid(x,y)
    
    lons = x*gt[1]+gt[0]
    lats = y*gt[5]+gt[3]
    
    return lats, lons

def create_df(array,keys):
    """
    Create a dataframe with a 3D numpy matrix.
    Each slice of the matrix (in 3rd dimension) is flattened into a linear
    vector and appended as a column to the dataframe.
    
    Parameters
    ----------
    array : 3d matrix of shape (rows, cols, features)
    keys : array of strings associated with each feature. This will be the 
            column name

    Returns
    -------
    df : pandas dataframe

    """
    df = pd.DataFrame()
    ctr=0
    for key in keys:
        df[key] = array[ctr].flatten()
        ctr+=1
    return df

def prettify_names(names):
    new_names = {"ks":"K$_{s,max}$",
                 "ndvi":"$NDVI_{mean}$",
                 "vpd_mean":"VPD$_{mean}$",
                 "vpd_cv":"VPD$_{CV}$",
                 "vpd_std":"VPD$_{std}$",
                 "thetas":"Soil porosity",
                 "elevation":"Elevation",
                 "dry_season_length":"Dry season length",
                 "ppt_mean":"Precip$_{mean}$",
                 "ppt_cv":"Precip$_{CV}$",
                 "agb":"Biomass",
                 "sand":"Sand %",
                 "clay":"Clay %",
                 "silt":"Silt %",
                 "canopy_height": "Canopy height",
                 "isohydricity":"Isohydricity",
                 "root_depth":"Root depth",
                 "hft":"Hydraulic\nfunctional type",
                 "p50":"$\psi_{50}$",
                 "gpmax":"$K_{max,x}$",
                 "c":"Capacitance",
                 "g1":"g$_1$",
                 "n":"$n$",
                 "bulk_density":"Bulk density",
                 "nlcd": "Land cover",
                 "nlcd_41.0": "Decid forest",
                 "nlcd_42.0": "Evergrn forest",
                 "nlcd_43.0": "Mixed forest",
                 "nlcd_52.0": "Shrub",
                 "nlcd_71.0": "Grass",
                 "nlcd_81.0": "Pasture",                 
                 "aspect":"Aspect",
                 "slope":"Slope",
                 "twi":"TWI",
                 "ppt_lte_100":"Dry months",
                 "dist_to_water":"Dist to water",
                 "t_mean":"Temp$_{mean}$",
                 "t_std":"Temp$_{st dev}$",
                 "lon":"Lon", "lat":"Lat",
                 "theta_third_bar": "$\psi_{0.3}$",
                 "vanGen_n":"van Genuchten n",
                 "AWS":"Avail water storage",
                 "AI":"Aridity Index",
                 "Sr": "RZ water storage",
                 "restrictive_depth": "Restricton depth",
                 "species":"species",
                 "basal_area": "Basal area",
                 "mnDFMC": "$DFMC_{mean}$",
                 "stdDFMC": "$DFMC_{std}$",
                 "HAND":"HAND",
                 "pws": "PWS"
                 }
    return [new_names[key] for key in names]

# Load data consistent with way it's done in rf_regression.py
pickleLoc = 'C:/repos/data/df_wSpec.pkl'
df_wSpec = pd.read_pickle(pickleLoc)

''' 
First make a separate figure for cross-correlations with DFMC values and climate
'''
#load mnDMFC
dirDFMC = 'C:/repos/data/DFMCstats/'
ds = gdal.Open(dirDFMC + "meanDFMC.tif")
gt = ds.GetGeoTransform()
mnDFMCMap = np.array(ds.GetRasterBand(1).ReadAsArray())
ds = None
lats, lons = get_lats_lons(mnDFMCMap, gt)
ds = gdal.Open(dirDFMC + "stdDFMC.tif")
stdDFMCMap = np.array(ds.GetRasterBand(1).ReadAsArray())
ds = None


#note that these are lat, lons of associated gri dcells with FIA sites in them, so should be at corners.
#can therefore round
latInd = np.round( (df_wSpec['lat'].to_numpy() - gt[3])/gt[5] ).astype(int)
lonInd = np.round( (df_wSpec['lon'].to_numpy() - gt[0])/gt[1] ).astype(int)
dfClim = df_wSpec.copy()
dfClim['mnDFMC'] = mnDFMCMap[latInd, lonInd]
dfClim['stdDFMC'] = stdDFMCMap[latInd, lonInd]
dfClim = dfClim[['mnDFMC','stdDFMC','vpd_mean','ppt_cv']]        

'''
for siteLat in df_wSpec['lat']:
    siteY, dist = 
for siteLon in df_wSpec['lon']:
    siteX = 
for (x, y) in zip(siteX, siteY):
    siteMnDFMC.append(mnDFMC[x, y])
    siteStdDFMC.append(stdDFMC[x, y])
'''

corrMatClim = dfClim.corr()
r2bcmap = sns.color_palette("RdBu", as_cmap=True)
fig, ax = plt.subplots(figsize = (3,3))
maskClim = np.full(corrMatClim.shape, False)
maskClim[2:,:] = True
sns.heatmap(dfClim.corr(), mask=maskClim,
        xticklabels=prettify_names(corrMatClim.columns.values),
        yticklabels=prettify_names(corrMatClim.columns.values),
        cmap = r2bcmap, vmin=-0.75, vmax=0.75,
        annot=True,  fmt=".2f", annot_kws={'size': 10})



'''
Make general cross-correlation map
'''
df_wSpec.drop(columns=['species','lat','lon'], inplace=True)
corrMat = df_wSpec.corr()
mask = np.triu(np.ones_like(corrMat, dtype=bool))
fig, ax = plt.subplots()
sns.heatmap(corrMat, mask=mask,
        xticklabels=prettify_names(corrMat.columns.values),
        yticklabels=prettify_names(corrMat.columns.values),
        cmap = r2bcmap, vmin=-0.75, vmax=0.75)
        #annot=True,  fmt=".1f", annot_kws={'size': 10})


