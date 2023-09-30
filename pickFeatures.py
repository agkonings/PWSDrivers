# -*- coding: utf-8 -*-
"""
Pick final features to be used in ML analysis, avoiding cross-correlation
Created on Mon Feb 20 17:13:30 2023

@author: konings
"""
import os

import numpy as np
import pandas as pd
from osgeo import gdal
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import dirs

#sns.set(font_scale = 1.2, style = "ticks")
#plt.rcParams.update({'font.size': 20})

#load features 
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

dfPath = os.path.join('C:/repos/data/inputFeatures_wgNATSGO_wBA_wHAND.h5')
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_allSeas_nonorm_4monthslag_exact6years.tif'
df =  load_data(dfPath, pwsPath)

#remove pixels with NLCD status that is not woody
df = df[df['nlcd']<70] #unique values are 41, 42, 43, 52
#remove mixed forest
df = df[df['nlcd'] != 43] #unique values are 41, 42, 43, 52

#drop traits, because that's too complicated to interpet/too coarse anyway
df.drop(['isohydricity','root_depth','p50','gpmax','c','g1','lat','lon'], axis = 1, inplace=True)
#drop old holdover things from Krishna's fire season work that don't apply year round or make sense for forests
specDropList = ['dry_season_length','ppt_lte_100','basal_area','nlcd','elevation']
df.drop(specDropList, axis = 1, inplace=True)
df.dropna(inplace = True)

#then calculate cross correation with PWS 
df.dropna(inplace=True) #remove points without species data/not at FIA
corrMat = df.corr()

error


#manually investigated as long as R<0.75
np.abs(corrMat['pws']).sort_values()
'''removed by manual inspection of cross-corr'
(vpd_mean, t_mean) = 0.84, remove t_mean,
(vpd_mean, vpd_cv) = 0.74 remove vpd_std
(slope, HAND) = 0.85 remove HAND
(AI, ppt_mean ) = 0.97
(ndvi, canopy_height) = 0.72
(ndvi, ppt_mean) = 0.73
(ndvi, agb) = 0.75
(clay, sand) = 0.81

NB: also looked at VPD-threemonthmax intead of vpd_mean (highly correlated)
but did worse in overall RF

then git say adjusted to have R< 0.6 metric for corss-correlation

'''
def plot_all_2Ddensities(df):
    columns = df.columns
    
    # Determine the number of rows and columns for subplots
    #assume 5 columns per row
    num_columns = 4
    num_rows = int( np.ceil(len(columns)/num_columns) )
    

    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 15))

    #calculate correlations to sort figures
    abs_pears = {}
    for i, column in enumerate(columns):
        if column != 'pws':             
            pearsonVal, _ = pearsonr(df['pws'], df[column])
            abs_pears[column] = np.abs( pearsonVal )
    
    #now re-arrange columns to go from highest to lowest Pearson
    corr_df = pd.DataFrame.from_dict(abs_pears, orient='index', columns=['Pearson'])
    sorted_columns = corr_df['Pearson'].sort_values(ascending=False).index

    # Iterate over each column and create a heatmap subplot    
    for i, column in enumerate(sorted_columns):
        
        if column != 'pws': 
            #re-calculate correlations
            pearson_corr, _ = pearsonr(df['pws'], df[column])    
            spearman_corr, _ = spearmanr(df['pws'], df[column])
            
            #plotting
            ax = axes[(i) // num_columns, (i) % num_columns]
            sns.kdeplot(data=df, x='pws', y=column, ax=ax, cmap='YlOrRd', fill=True)
            ax.set_xlim(0, 1.6)
            if column == 'ks':
                ax.set_ylim(0, 150)
            text = f"$R_{{Pears}}$: {pearson_corr:.2f}\n$R_{{Spear}}$: {spearman_corr:.2f}"
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top')

        
    fig.tight_layout()

    # Display the figure
    plt.show()

#plot_all_2Ddensities(df_wSpec)


'''
go back and look at climate correlation matrices only
'''
#first create climate matrix with lots of extra variables to cut down to only pixels with FIA plots and traits
dfAll =  load_data(dfPath, pwsPath)
dfClim = dfAll[['elevation','pws','ndvi','vpd_mean','vpd_cv','ppt_mean','ppt_cv','t_mean','t_std','g1','basal_area','agb','elevation']]
dfClim.dropna(inplace=True)
dfClim.drop(columns=['g1','basal_area'], inplace=True) #then drop irrelevant columns
dfClim['vpd_std'] = df['vpd_cv']*df['vpd_mean']
dfClim['ppt_std'] = df['ppt_cv']*df['ppt_mean']

#plot correlation matrix in two colormaps
corrMatClim = dfClim.corr()
r2bcmap = sns.color_palette("vlag", as_cmap=True)
fig, ax = plt.subplots()
sns.heatmap(corrMatClim, 
            cmap = r2bcmap, vmin=-1, vmax=1)
fig, ax = plt.subplots()
sns.heatmap(np.abs(corrMatClim),
            vmin=0, vmax=1)

#then consider vod_std and ppt_std. Where the CVs the way to go?
#is VPDcv signicantly less correlated with VPDmean than or NDVI than VPDstd?
#no -> VPDcv and VPDstd both ~0.75 with vpd_mean 
#is pptcv signicantly less correlated with pptmean than or NDVI than pptstd?
#YES - > pptCV is 0.05, whereas ppt_std = 0.91
#what is correlation with TWImean and TWImean
#then decide: do we really need all 5 climate? Can we experiment with dropping any and doing about as well?

#see what biomass, vpd_mean, ppt_mean, ppt_cv, vpd_std gives
#how muc hR2 reduction if drop lowest of those? and do three traits each per type?
 
