# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:10:13 2021

@author: kkrao

Create dataframe of with input (features) and output (pws)
Shape of dataframe = (# of pixels, # of features + 1)
"""

import os

import pandas as pd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dirs

sns.set(font_scale = 0.6, style = "ticks")


def get_value(filename, band = 1):
    """
    Vector implementation of query of raster value at lats and lons

    Parameters
    ----------
    filename : raster path    
    band : band position to query. int, optional. The default is 1.

    Returns
    -------
    1D array of value of raster at lats and lons

    """
    ds = gdal.Open(filename)
    data = ds.GetRasterBand(band).ReadAsArray()    
    ds = None
    return data



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

def check_regridding(gtPWS, gtShape, featFiles, regridDir):    
    for file in featFiles:
        filepath = os.path.join(regridDir, file)
        ds = gdal.Open(filepath)
        gt = ds.GetGeoTransform()
        fileData = ds.GetRasterBand(1).ReadAsArray()
        if fileData.shape != gtShape or gt != gtPWS:
            print('checked ' + file + ': WARNING HAS PROBLEM')
            if fileData.shape != gtShape:
                print('checked ' + file + ': shape HAS PROBLEM')
            if gt != gtPWS:   
                print('checked ' + file + ': geotransform HAS PROBLEM')
                print('gtPWS: ' )
                print(gtPWS)
                print('gt file: ')
                print(gt)
        else: 
            print('checked ' + file + ': all good') 

def create_h5(store_path):
    """
    Bring together all features and labels (pws). drop nans. Store it as a h5
    file. And return it as a dataframe of shape (# of examples, # of features + 1)
                                                 
    Returns
    -------
    df : pandas dataframe

    """
    data = dict()
    dsPWS = gdal.Open(os.path.join(dirs.dir_data, "pws_features","PWS_through2021.tif"))
    gtPWS = dsPWS.GetGeoTransform()
    data['pws'] = np.array(dsPWS.GetRasterBand(1).ReadAsArray())
    lats, lons = get_lats_lons(data['pws'], gtPWS)
    
    regridDir = 'G:/My Drive/0000WorkComputer/dataStanford/PWSFeatures/resampled'
    featFiles = os.listdir(regridDir)
    
    #check that all files are now regridded properly with same geotransform and
    #shape. Note that land cover file has problems. Just don't load for now
    check_regridding(gtPWS, data['pws'].shape, featFiles, regridDir)    
    
    keys = ['sand','clay','ks','bulk_density','theta_third_bar','isohydricity',\
        'root_depth','canopy_height','p50','gpmax', 'c','g1','nlcd',
        "elevation","aspect","slope","twi","dry_season_length","ndvi",\
            "vpd_mean","vpd_cv", "dist_to_water","agb","ppt_mean","ppt_cv",\
        "t_mean","t_cv","ppt_lte_100", "AI","Sr","AWS", "restrictive_depth", "species", "basal_area", "lon","lat","HAND"]
    
    array = np.zeros((len(keys), data['pws'].shape[0],data['pws'].shape[1])).astype('float')
    
    
    #add data one by one
    array[0] = get_value( os.path.join(regridDir, 'SandPercent_0to150cm_4km_westernUS.tif'), 1)
    array[1] = get_value( os.path.join(regridDir, 'ClayPercent_0to150cm_4km_westernUS.tif'), 1)
    array[2] = get_value( os.path.join(regridDir, 'Ksat_0to50cm_4km_westernUS.tif'), 1)
    array[3] = get_value( os.path.join(regridDir, 'BulkDensityOneThirdBar_0to50cm_4km_westernUS.tif'), 1)
    array[4] = get_value( os.path.join(regridDir, 'WaterContentOneThirdBar_0to50cm_4km_westernUS.tif'), 1)    
    array[5] = get_value( os.path.join(regridDir, 'isohydricity.tif'), 1)
    array[6] = get_value( os.path.join(regridDir, 'root_depth.tif'), 1)
    array[7] = get_value( os.path.join(regridDir, 'canopy_height.tif'), 1)
    array[8] = get_value( os.path.join(regridDir, 'P50_liu.tif'), 1)
    array[9] = get_value( os.path.join(regridDir, 'gpmax_50.tif'), 1)
    array[10] = get_value( os.path.join(regridDir, 'C_50.tif'), 1)
    array[11] = get_value( os.path.join(regridDir, 'g1_50.tif'), 1)
    array[12] = get_value( os.path.join(regridDir, 'nlcd_2016_4km.tif'), 1)    
    array[13] = get_value( os.path.join(regridDir, 'usa_dem.tif'), 1)    
    array[14] = get_value( os.path.join(regridDir, 'usa_aspect_wgs1984_clip.tif'), 1)    
    array[15] = get_value( os.path.join(regridDir, 'usa_slope_project.tif'), 1)    
    #array[16] = get_value( os.path.join(regridDir, 'twi.tif'), 1)    
    array[16] = get_value( os.path.join(regridDir, 'twi_epsg4326_4000m_merithydro.tif'), 1)    
    array[17] = get_value( os.path.join(regridDir, 'fireSeasonLength.tif'), 1)    
    array[18] = get_value( os.path.join(regridDir, 'ndvi_mean.tif'), 1)    
    array[19] = get_value( os.path.join(regridDir, 'vpd_mean.tif'), 1)    
    #array[20] = get_value( os.path.join(regridDir, 'vpdStd.tif'), 1)        
    vpdStd = get_value( os.path.join(regridDir, 'vpdStd.tif'), 1)        
    array[20] = vpdStd/array[19]
    array[21] = get_value( os.path.join(regridDir, 'distance_to_water_bodies.tif'), 1)    
    array[22] = get_value( os.path.join(regridDir, 'agb_2020.tif'), 1)    
    array[23] = get_value( os.path.join(regridDir, 'pptMean.tif'), 1)        
    pptStd = get_value( os.path.join(regridDir, 'pptStd.tif'), 1)
    array[24] = pptStd/array[23]  
    #note different bands due to bug in file creation
    array[25] = get_value( os.path.join(regridDir, 'tMean.tif'), 2)    
    tStd = get_value( os.path.join(regridDir, 'tStd.tif'), 2)    
    array[26] = tStd/array[25]
    array[27] = get_value( os.path.join(regridDir, 'ppt_lte_100.tif'), 1)    
    array[28] = get_value( os.path.join(regridDir, 'aridity_index.tif'), 1)
    array[29] = get_value( os.path.join(regridDir, 'Sr_2020_unmasked_4km_westernUS.tif'), 1)    
    array[30] = get_value( os.path.join(regridDir, 'AWS_0to150cm_4km_westernUS.tif'), 1)
    array[31] = get_value( os.path.join(regridDir, 'RestrictiveLayerDepth_resampled_clipped.tif'), 1)                
    array[32] = get_value( 'C:/repos/data/FIADomSpecies.tif', 1)
    array[33] = get_value( os.path.join(regridDir, 'FIABasalAreaAc.tif'), 1)        
    array[34] = get_value( 'C:/repos/data/FIALons.tif', 1)
    array[35] = get_value( 'C:/repos/data/FIALats.tif', 1)  
    array[36] = get_value( os.path.join(regridDir, 'hand_epsg4326_4000m_merithydro.tif'), 1)    
    
    
    ds = None
    
    df = create_df(array,keys)
    #df.dropna(subset = ["pws"], inplace = True)
    print(df.describe())
    print(df.head())
    
    df.describe()
    df.loc[df['sand']<-1] = np.nan
    df.loc[df['clay']<-1] = np.nan
    df.loc[df['ks']<-1] = np.nan
    df.loc[df['bulk_density']<0] = np.nan
    df.loc[df['nlcd']<41] = np.nan #remove open space & developed land
    df.loc[df['nlcd']>81] = np.nan #renive crops
    df.loc[df['elevation']<-1e3] = np.nan
    df.loc[df['slope']<-1e3] = np.nan
    df.loc[df['aspect']>2e3] = np.nan
    df.loc[df['twi']>2e3] = np.nan
    df.loc[df['restrictive_depth']>1000] = np.nan
    df.loc[df['restrictive_depth']<0] = np.nan
    df.loc[df['theta_third_bar']<0] = np.nan
    df.loc[df['AWS']<0] = np.nan
    
    '''
    #plot map of where there is data to debug
    #first load pws to get grid size
    filename = os.path.join("C:/repos/data/pws_features/PWS_through2021_allSeas.tif") #load an old PWS file. 
    ds = gdal.Open(filename)
    geotransform = ds.GetGeoTransform()
    pws = np.array(ds.GetRasterBand(1).ReadAsArray())
    #plot map
    df2 = df.copy()
    droppedFeats = ['sand', 'silt', 'ks', 'nlcd', 'bulk_density','theta_third_bar',
                    'isohydricity','root_depth','p50','gpmax','c','g1',
                   'elevation', 'slope', 'aspect', 'twi']
    df2.dropna(subset = droppedFeats, inplace = True)
    latMap = np.empty( np.shape(pws) ) * np.nan
    latInd = np.round( (df2['lat'].to_numpy() - geotransform[3])/geotransform[5] ).astype(int)
    lonInd = np.round( (df2['lon'].to_numpy() - geotransform[0])/geotransform[1] ).astype(int)
    latMap[latInd, lonInd] = 1
    fig, ax1 = plt.subplots()
    im = ax1.imshow(latMap, interpolation='none')
    plt.title('lats with PWS, soil, elev, and PFT, NaNs')
    print('removing PWS, soil, elev, and PFT NaNs has length: ' + str(len(df2)))
    
    df.loc[df['isohydricity']>1e3] = np.nan
    df.loc[df['root_depth']<-1] = np.nan
    df.loc[df['p50']<-1e3] = np.nan
    df.loc[df['gpmax']<-1e3] = np.nan
    df.loc[df['c']<-1e3] = np.nan
    df.loc[df['g1']<-1e3] = np.nan

    #plot map after traits
    df3 = df.copy()
    df3.dropna(inplace = True)
    latMap = np.empty( np.shape(pws) ) * np.nan
    latInd = np.round( (df3['lat'].to_numpy() - geotransform[3])/geotransform[5] ).astype(int)
    lonInd = np.round( (df3['lon'].to_numpy() - geotransform[0])/geotransform[1] ).astype(int)
    latMap[latInd, lonInd] = 1
    fig, ax1 = plt.subplots()
    im = ax1.imshow(latMap, interpolation='none')
    plt.title('lats with PWS, soil, elev, PFT, and trait, NaNs')
    print('removing all NaNs has length: ' + str(len(df3)))
    '''
    

    store = pd.HDFStore(store_path)
    store['df'] = df
    store.close()
    
    return df

#%%load dataset

def plot_heatmap(df):
    """
    Make a heatmap of correlations between PWS and features.
    """
    sample = df.sample(10000)
    sns.pairplot(sample)
    
    
    sns.heatmap(df.corr(),vmin = -0.2, vmax = 0.2, cmap = sns.diverging_palette(240, 10, n=8))
    
    fig, axs= plt.subplots(5,3,figsize = (8,8),sharey=True)
    axs = axs.flatten()
    ctr=0
    for col in df.columns:
        if col=="pws":
            continue
        # df[col].hist()
        # ax.set_xlabel(col)
        axs[ctr].set_ylim(-0.1,2)
        if col=="hft":
            sns.boxplot(x=col, y="pws", data=df.sample(int(1e4)),ax=axs[ctr],color = "grey")
        else:
            # sns.regplot(x=col, y="pws", data=df.sample(int(1e4)),ci = None,
                          # scatter_kws={"s": 0.1,'alpha':0.5},line_kws = {"linewidth":0},ax=axs[ctr],color = "grey")
            sns.kdeplot(x=col, y="pws", data=df.sample(int(1e4)),fill=True,
                          cmap = "Greys",ax=axs[ctr])
            axs[ctr].set_xlim(df[col].quantile(0.05),df[col].quantile(0.95))
        axs[ctr].set_xlabel("")
        axs[ctr].annotate("%s"%str(col), xy=(0.5, 0.9), xytext=(0.5, 0.95), ha = "center", va = "top", textcoords="axes fraction",fontsize = 10)
        ctr+=1
    plt.show()
    return axs

def main():
    #%% make and save dataframe:
    store_path = os.path.join(dirs.dir_data, 'inputFeatures_wgNATSGO_wBA_wHAND.h5')
    create_h5(store_path)
    
    #%% Load h5 as type
    # make sure dirs.dir_data in dirs.py points to location of store_plant_soil_topo_climate.h5
    # This is typically location of repo/data
    store = pd.HDFStore(store_path)
    df =  store['df']
    store.close()
    df.columns = df.columns.astype(str)    
    df.head()
    
        
if __name__ == "__main__":
    main()