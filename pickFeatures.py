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

dfPath = os.path.join('C:/repos/data/inputFeatures_wgNATSGO_wBA.h5')
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWS_through2021.tif'
df =  load_data(dfPath, pwsPath)

#find data with valid PWS
for feat in df.drop('pws', axis=1).columns:
    nMissing = np.sum( ~np.isnan(df[feat]) & ~np.isnan(df['pws']) )
    print('Valid data for feat ' + str(feat) + ': ' + str(nMissing))                        

#figure out how much overlap, what are the big losses?
#Big loses: P50, gpmax, g1, species, Sr
df.drop(['isohydricity','root_depth','p50','gpmax','c','g1','species','lat','lon'], axis = 1, inplace=True)

#then calculate cross correation with PWS 
corrMat = df.corr(method='spearman')

'''
for simplicity, just manually look at what is highest wth PWS
allow nothing corss-correalted greater than 0.5
keep VPD_mean, ndvi, 
then VPD_CV too cross-correlated, t_mean too cross-correlated, canopy_height too, dry_season_length
drop AI because corss-corr with VPD_min by 0.63. controversial
etc....two ks, ppt_cv with only slightly above 0.5 correlation.
so with threshold of 0.5 hard cut-off what is left is
VPD_min, NDVI, bulk_density, dist_to_water, nlcd, elevation, sand, clay
'''
'''
If redo with spearman
vpd_mean, ndvi, aws, theta_third_bar, ppt_cv, restrictive_depth,
sand, clay, ks, bulk_density, theta_third_bar,'nlcd','elevation', 'aspect','twi'
dist_to_water

'''

'''
want to repeat with looking at what is cross-correlated with speices
and where have data that overlaps with species 
'''

#calculate with all dominant species
df_wSpec =  load_data(dfPath, pwsPath)
df_wSpec.drop(['elevation','isohydricity','root_depth','p50','gpmax','c','g1','species','lat','lon'], axis = 1, inplace=True)
df_wSpec.dropna(inplace = True)

corrMatSpec = df_wSpec.corr()
#manually investigated as long as R<0.75
specDropList = ['dry_season_length','t_mean','AI','ppt_lte_100']
df_wSpec.drop(specDropList, axis = 1, inplace=True)
df_wSpec.dropna(inplace = True)



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

plot_all_2Ddensities(df_wSpec)