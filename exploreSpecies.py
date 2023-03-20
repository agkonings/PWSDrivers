# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from osgeo import gdal
import sklearn.ensemble
import sklearn.model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
import matplotlib.patches
import sklearn.inspection
from sklearn.inspection import permutation_importance
import sklearn.metrics
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr

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

def get_categories_and_colors():
    """
    colors and categorize to combine feature importance chart
    """
    
    green = "yellowgreen"
    brown = "saddlebrown"
    blue = "dodgerblue"
    yellow = "khaki"
    purple = "magenta"
    
    plant = ['canopy_height', "agb",'ndvi', "nlcd","species"]
    soil = ['clay', 'sand','silt','thetas', 'ks', 'vanGen_n','Sr','Sbedrock','bulk_density','theta_third_bar','AWS']
    climate = [ 'dry_season_length', 'vpd_mean', 'vpd_cv',"ppt_mean","ppt_cv","t_mean","t_std","ppt_lte_100","AI"]
    topo = ['elevation', 'aspect', 'slope', 'twi',"dist_to_water"]
    traits = ['isohydricity', 'root_depth', 'p50', 'gpmax', 'c', 'g1']
    
    return green, brown, blue, yellow, purple, plant, soil, climate, topo, traits

def prettify_names(names):
    new_names = {"ks":"K$_{s,max}$",
                 "pws": "PWS",
                 "ndvi":"NDVI",
                 "vpd_mean":"VPD$_{mean}$",
                 "vpd_cv":"VPD$_{CV}$",
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
                 "species_64.0":"Species 64", "species_65.0":"Species 65",
                 "species_69.0":"Species 69", "species_106.0":"Species 106",
                 "species_108.0":"Species 108", "species_122.0":"Species 122",
                 "species_133.0":"Species 133", "species_202.0":"Species 202",
                 "species_746.0":"Species 746", "species_756.0":"Species 756",
                 "species_814.0":"Species 814"
                 }
    return [new_names[key] for key in names]
        
def plot_corr_feats(df):
    '''
    Plot of feature correlation to figure out what to drop
    takes in dataframe
    returns axis handle

    '''
    X = df.drop("pws",axis = 1)
    corrMat = X.corr()
    r2bcmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(corrMat, 
            xticklabels=prettify_names(corrMat.columns.values),
            yticklabels=prettify_names(corrMat.columns.values),
            cmap = r2bcmap, vmin=-0.75, vmax=0.75)

def plot_preds_actual(X_test, y_test, regrn, score):
    """
    Plot of predictions vs actual data
    """
    y_hat =regrn.predict(X_test)
    
    fig, ax = plt.subplots(figsize = (3,3))
    ax.scatter(y_hat, y_test, s = 1, alpha = 0.05, color='k')
    ax.set_xlabel("Predicted PWS", fontsize = 18)
    ax.set_ylabel("Actual PWS", fontsize = 18)
    ax.set_xlim(0,1.6)
    ax.set_ylim(0,1.6)
    ax.annotate(f"R$^2$={score:0.2f}", (0.1,0.9),xycoords = "axes fraction", ha = "left")
    return ax    
    
    
plt.rcParams.update({'font.size': 18})


#%% Load data
dfPath = os.path.join(dirs.dir_data, 'inputFeatures_wgNATSGO_wBA.h5')
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWS_through2021_JunThruNov.tif'
df =  load_data(dfPath, pwsPath)
#sdroppedvarslist based on manual inspection so no cross-correlations greater than 0.5, see pickFeatures.py
#droppedVarsList = ['lat','lon','species','restrictive_depth','lat','lon','slope','vpd_cv','t_mean','canopy_height','dry_season_length','vpd_mean','ppt_mean','t_std','ppt_lte_100','agb'] #,'root_depth','ks','ppt_cv']
specDropList = ['lat','lon','dry_season_length','vpd_cv','canopy_height','ppt_mean','ppt_lte_100','agb']
droppedVarsList = specDropList
df = cleanup_data(df, droppedVarsList)

#common species list hand-calculated separately based on most entries in species column
#shorter list gets about 0.03 better r2 because have more data, so use that as in between point
#between enough data to do well and not too many columns
commonSpecList = {65, 69, 122, 202, 756, 64, 106, 108}

#filter so that only data with most common species are kept
noDataRows = df.loc[~df.species.isin(commonSpecList)]
df.drop(noDataRows.index, inplace=True)

namesDict = prettify_names(df.columns.values)
#plot cross-correlations
colors = {'k','tab:blue', 'tab:orange', 'tab:green', 'tab:red', \
          'tab:purple', 'tab:brown', 'tab:pink'}
count = pd.DataFrame()
binCenters = pd.DataFrame()
for var in df.columns:
    fig, ax = plt.subplots()
    for thisSpec in commonSpecList:
        if var == 'AI':
            binEdges = np.concatenate((np.arange(0,1,0.1), np.arange(1,3.2,0.2)), axis=0)
            count[thisSpec], binEdges = np.histogram(df[var].loc[df.species==thisSpec], binEdges)    
        else:
            count[thisSpec], binEdges = np.histogram(df[var].loc[df.species==thisSpec], 20)    
        binCenters[thisSpec] = ( binEdges[0:-1] + binEdges[1:] ) / 2
    plt.xlabel( prettify_names([var])[0] )
    plt.plot(binCenters, count) 
    #plt.legend(['Western junip', 'Utah junip','Oneseed junip','Pinyon pine','Lodgepole','Ponderosa pine','Douglas fir','Honey mesquite'])
    
 