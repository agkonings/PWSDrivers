# -*- coding: utf-8 -*-
'''
Look for 4km PWS pixels where there are multiple FIA plots with a dominant
species. Do they have the same dominant species?''

...turns out there are no 4 km pixels with multiple FIA sites. 
Maybe should have been able to predict that? Ah well
'''

import os

import numpy as np
import pandas as pd
from osgeo import gdal
import sklearn.ensemble
import sklearn.model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_scatter_density
import sklearn.preprocessing
import matplotlib.patches
import sklearn.inspection
from sklearn.inspection import permutation_importance
import sklearn.metrics
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
import pickle

import dirs

sns.set(font_scale = 1, style = "ticks")
plt.rcParams.update({'font.size': 18})


def add_pws(df, pwsPath):
    '''
    add pws to data frame from a particular path
    Here, we assume that the lats, lons, and other variables in the dataframe
    have the same original 2-D shape as the PWS array, and that each 1-D version
    in the dataframe is created by .flatten(). For more info on how this is
    done in inputFeats dataframes, see make_data.py
    '''
    
    dspws = gdal.Open(pwsPath)
    gtpws= dspws.GetGeoTransform()
    arraypws = np.array(dspws.GetRasterBand(1).ReadAsArray())
    
    df['pws'] = arraypws.flatten()
    
    #re-arrange columns so pws goes first
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    return df

def load_data(dfPath, pwsPath):
    '''
    create a joint dataframe with the input features + the pws 
    '''    

    store = pd.HDFStore(dfPath)
    df =  store['df']   # save it
    store.close()

    #add particular PWS file
    df = add_pws(df, pwsPath)

    return df

def cleanup_data(df, droppedVarsList, filterList=None):
    """
    path : where h5 file is stored
    droppedVarsList : column names that shouldn't be included in the calculation
    """
    
    df.drop(droppedVarsList, axis = 1, inplace = True)
    df.dropna(inplace = True)
        
    df.reset_index(inplace = True, drop = True)    
    return df
    
    
plt.rcParams.update({'font.size': 18})


#%% Load data
dfPath = os.path.join(dirs.dir_data, 'inputFeatures_wgNATSGO_wBA_wHAND.h5')
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWS_through2021_allSeas_4monthslag.tif'
df_wSpec =  load_data(dfPath, pwsPath)


'''Process ancillary variables exactly as in main regression so looking at same exact plots'''
#sdroppedvarslist based on manual inspection so no cross-correlations greater than 0.75, see pickFeatures.py
#further added nlcd to drop list since doesn't really make sense if focusing on fia plots
specDropList = ['dry_season_length','t_mean','AI','t_std','ppt_lte_100','elevation']
cleanlinessDropList = ['HAND','restrictive_depth','canopy_height','Sr','root_depth','bulk_density']
droppedVarsList = specDropList + cleanlinessDropList + ['ppt_mean','agb','theta_third_bar','clay','vpd_std','basal_area','dist_to_water','p50','gpmax']
droppedVarsList.remove('ppt_mean')
droppedVarsList.remove('vpd_std')
df_wSpec = cleanup_data(df_wSpec, droppedVarsList)

#remove pixels with NLCD status that is not woody
df_wSpec = df_wSpec[df_wSpec['nlcd']<70] #unique values are 41, 42, 43, 52

#calculate which 4 km pixel in
ds = gdal.Open(pwsPath)
gt = ds.GetGeoTransform()
indX = np.floor( (df_wSpec['lon']-gt[0])/gt[1] ).to_numpy().astype(int)
indY = np.floor( (df_wSpec['lat']-gt[3])/gt[5] ).to_numpy().astype(int)
pixelIndex = indX*10000 + indY
df_wSpec['pixelIndex'] = pixelIndex
np.max(df_wSpec['pixelIndex'].value_counts())


