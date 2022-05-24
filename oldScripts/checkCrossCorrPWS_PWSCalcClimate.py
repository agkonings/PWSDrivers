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

dirData = 'C:/repos/data/pws_features/'

#calculate lats and lons lists in PWS bounds
data = dict()
ds = gdal.Open(dirData + "PWS_through2021.tif")
gt = ds.GetGeoTransform()
data['pws'] = np.array(ds.GetRasterBand(1).ReadAsArray())
lats, lons = get_lats_lons(data['pws'], gt)

#create data structures for eventual pandas df
keys = ["mnDFMC", "stdDFMC", "q5DFMC", "q10DFMC", "mnVPD", "stdVPD", "mnPrec", "stdPrec", "mnTemp", "stdTemp"]
array = np.zeros((len(keys), data['pws'].shape[0],data['pws'].shape[1])).astype('float')

#Get everything in same gridding format
#code structure heavily copied from make_data.py
array[0] = get_values( dirData +  "meanDFMC.tif" , lons, lats)
array[1] = get_values( dirData +  "stdDFMC.tif" , lons, lats)
array[2] = get_values( dirData +  "q5DFMC.tif", lons, lats)
array[3] = get_values( dirData +  "q10DFMC.tif", lons, lats)

#Get climate data 
ds = gdal.Open( dirData + "vpd_mean.tif")
array[4] = ds.GetRasterBand(1).ReadAsArray()
ds = gdal.Open( dirData + "vpdStd.tif")
array[5] = ds.GetRasterBand(1).ReadAsArray()
ds = gdal.Open( dirData + "pptMean.tif")
array[6] = ds.GetRasterBand(1).ReadAsArray()
array[8] = ds.GetRasterBand(2).ReadAsArray()
ds = gdal.Open( dirData + "pptStd.tif")
array[7] = ds.GetRasterBand(1).ReadAsArray()
array[9] = ds.GetRasterBand(2).ReadAsArray()

#create pandas data frame  
df = create_df(array, keys)
df.dropna(inplace = True)

#plot cross-correlation matrix
corrMat = df.corr()
r2bcmap = sns.color_palette("vlag", as_cmap=True)
sns.heatmap(corrMat, 
        xticklabels=corrMat.columns.values,
        yticklabels=corrMat.columns.values,
        cmap = r2bcmap, vmin=-0.65, vmax=0.65)

#result: not much cross-corr that could explain why mean VPD is such a big influence!
#-0.27 between mean VPD and std DFMC is biggest crosscorr
#so if mean VPD is larger, DFMC is less variable