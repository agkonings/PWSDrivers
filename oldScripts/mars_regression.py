# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:32:44 2022

@author: konings
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import sklearn.model_selection 
from pyearth import Earth
import matplotlib.pyplot as plt

#functions on colorizing, names
def get_categories_and_colors():
    """
    colors and categorize to combine feature importance chart
    """
    
    green = "yellowgreen"
    brown = "saddlebrown"
    blue = "dodgerblue"
    yellow = "khaki"
    purple = "magenta"
    
    plant = ['canopy_height', "agb",'ndvi', "lc","pft"]
    soil = ['sand',  'clay', 'silt','thetas', 'ks', 'vanGen_n']
    climate = [ 'dry_season_length', 'vpd_mean', 'vpd_cv',"ppt_mean","ppt_cv","t_mean","t_std","ppt_lte_100"]
    topo = ['elevation', 'aspect', 'slope', 'twi',"dist_to_water"]
    traits = ['isohydricity', 'root_depth', 'hft', 'p50', 'gpmax', 'c', 'g1']
    
    return green, brown, blue, yellow, purple, plant, soil, climate, topo, traits

def prettify_names(names):
    new_names = {"ks":"K$_{s,max}$",
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
                 "gpmax":"Max. xylem\nconductance",
                 "c":"Xylem\ncapacitance",
                 "g1":"g$_1$",
                 "n":"$n$",
                 "pft":"Plant Functional Type",
                 "aspect":"Aspect",
                 "slope":"Slope",
                 "twi":"TWI",
                 "ppt_lte_100":"Dry months",
                 "dist_to_water":"Dist to water",
                 "t_mean":"Temp$_{mean}$",
                 "t_std":"Temp$_{st dev}$",
                 "lon":"Lon",
                 "lat":"Lat",
                 "vanGen_n":"van Genuchten n"
                 }
    return [new_names[key] for key in names]
    

#load data and clean-up
path = '../data/store_plant_soil_topo_climate_PWSthrough2021v3.h5'
store = pd.HDFStore(path)
df =  store['df']   # save it
store.close()
#remove NaNs, cross-correlated variables
#df.drop(["lc","ndvi","hft","sand",'vpd_cv',"ppt_lte_100","thetas","dry_season_length","t_mean","t_std","lat","lon"],axis = 1, inplace = True)
df.drop(["lc","vpd_mean","vpd_cv","ppt_mean","ppt_cv","ndvi","hft","sand",'vpd_cv',"ppt_lte_100","thetas","dry_season_length","t_mean","t_std","lat","lon"],axis = 1, inplace = True)
df.dropna(inplace = True)
df.reset_index(inplace = True, drop = True)
corrMatSp = df.corr(method='spearman')
error

#create initial model
model = Earth(max_degree=1, feature_importance_type='gcv', verbose=True)
X = df.drop("pws",axis = 1)
y = df['pws']
model.fit(X, y)
y_hat = model.predict(X)
scoreR2 = np.corrcoef(y, y_hat)

#[experiment with cost function?]
#consider R2 as comparison

#plot improtance to get a sense. 
imp = model.feature_importances_
def plot_importance(imp):
    """
    plot feature importance for all features

    Parameters
    ----------
    imp : dataframe returned by regress

    Returns
    -------
    ax: axis handle

    """
    
    fig, ax = plt.subplots(figsize = (5.5,7))
    green, brown, blue, yellow, purple, plant, soil, climate, topo, traits \
                                            = get_categories_and_colors()

    imp.plot.barh(y = "importance",x="symbol",color = imp.color, edgecolor = "grey", ax = ax, fontsize = 18)

    legend_elements = [matplotlib.patches.Patch(facecolor=green, edgecolor='grey',
                             label='Plant'), 
                       matplotlib.patches.Patch(facecolor=brown, edgecolor='grey',
                             label='Soil'), 
                       matplotlib.patches.Patch(facecolor=yellow, edgecolor='grey',
                             label='Topography'), 
                       matplotlib.patches.Patch(facecolor=blue, edgecolor='grey',
                             label='Climate'),
                       matplotlib.patches.Patch(facecolor=purple, edgecolor='grey',
                             label='Traits')]
    ax.legend(handles=legend_elements, fontsize = 18)
    ax.set_xlabel("Variable importance", fontsize = 18)
    ax.set_ylabel("")
    ax.set_xlim(0,0.60)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    return ax


# yhat = model.predict([row])
