# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 07:44:09 2021

@author: kkrao
random forest regression of pws
"""
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
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent, show
import shap
import time
import pickle

import dirs

sns.set(font_scale = 1, style = "ticks")
plt.rcParams.update({'font.size': 18})

def cleanup_data(path):
    """
    path is where h5 file is stored
    """
    
    store = pd.HDFStore(path)
    df =  store['df']   # save it
    store.close()  
    #df.drop(["lc","isohydricity",'root_depth', 'hft', 'p50', 'c', 'g1',"dry_season_length","lat","lon"],axis = 1, inplace = True)
    df.drop(["lc","vpd_mean","vpd_cv","ppt_mean","ppt_cv","ndvi","hft","sand",'vpd_cv',"ppt_lte_100","thetas","dry_season_length","t_mean","t_std"],axis = 1, inplace = True)
    #df.drop(["lc","ndvi","dry_season_length","lat","lon"],axis = 1, inplace = True)
    df.dropna(inplace = True)
    lat = df["lat"]
    lon = df["lon"]
    df.drop(["lat", "lon"], axis=1, inplace=True)
    df.reset_index(inplace = True, drop = True)
    
    return df, lat, lon


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
    
def get_shap_categories():
    """
    categorize for tractability
    """
    
    plantShaps = ['shap_canopy_height', 'shap_pft', 'shap_agb']
    soilShaps = ['shap_silt', 'shap_clay', 'shap_ks', 'shap_vanGen_n']
    topoShaps = ['shap_elevation', 'shap_twi', 'shap_dist_to_water']
    traitsShaps = ['shap_isohydricity', 'shap_root_depth', 'shap_p50', 'shap_gpmax', 'shap_c', 'shap_g1']
    
    return plantShaps, soilShaps, topoShaps, traitsShaps

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
                 "root_depth":"Rooting depth",
                 "hft":"Hydraulic\nfunctional type",
                 "p50":"$\psi_{50}$",
                 "gpmax":"K_{xylem,max}",
                 "c": "capacitance",
                 "g1":"g$_1$",
                 "n":"$n$",
                 "pft":"PFT",
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
    
    
    
def regress(df):
    """
    Regress features on PWS using rf model
    Parameters
    ----------
    df : columns should have pws and features

    Returns:
        X_test:dataframe of test set features
        y_test: Series of test set pws
        regrn: trained rf model (sklearn)
        imp: dataframe of feature importance in descending order
    -------
    

    """
    # separate data into features and labels
    X = df.drop("pws",axis = 1)
    y = df['pws']
    # separate into train and test set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.33, random_state=32)
    '''
    # Checking if leaves or node_impurity affects performance
    # after running found that it has almost no effect (R2 varies by 0.01)
    for leaves in [3, 4, 6]: #[6,7,8,9,10,12, 14, 15]:
        for decrease in [ 1e-8, 1e-9,5e-10,1e-10]:
            for nEst in [50,90,120,140]:
                # construct rf model
                regrn = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                              min_impurity_decrease=decrease, n_estimators = nEst)
                # train
                regrn.fit(X_train, y_train)
                # test set performance
                score = regrn.score(X_test,y_test)
                print(f"[INFO] score={score:0.3f}, leaves={leaves}, decrease={decrease}, nEst = {nEst}")
    # choose min leaves in terminal node and node impurity
    ''' 
    #can get highest with 3 leaves, 120 nEst, decrease 1e-8, but that seems like low number of leaves
    #old configuration was leaves = 6, decrease 1e-6, nEst = 50
    leaves = 4
    decrease = 1e-8
    # construct rf model
    regrn = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                      min_impurity_decrease=decrease, n_estimators = 90)
    # train
    regrn.fit(X_train, y_train)
    # test set performance
    score = regrn.score(X_test,y_test)
    print(f"[INFO] score={score:0.3f}, leaves={leaves}, decrease={decrease}")
    
    # assemble all importance with feature names and colors
    rImp = permutation_importance(regrn, X_test, y_test,
                            n_repeats=2, random_state=0)
    heights = rImp.importances_mean
    #heights = regrn.feature_importances_
    ticks = X.columns
  
    green, brown, blue, yellow, purple, plant, soil, climate, topo, traits \
                                            = get_categories_and_colors()
    
    imp = pd.DataFrame(index = ticks, columns = ["importance"], data = heights)
    
    def _colorize(x):
        if x in plant:
            return green
        elif x in soil:
            return brown
        elif x in climate:
            return blue
        elif x in traits:
            return purple
        else:
            return yellow
    imp["color"] = imp.index
    imp.color = imp.color.apply(_colorize)
    imp["symbol"] = imp.index
    # cleanup variable names
    imp.symbol = prettify_names(imp.symbol)
    imp.sort_values("importance", ascending = True, inplace = True)
    print(imp.groupby("color").sum().round(2))

    return X_test, y_test, regrn, score, imp


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
    ax.scatter(y_hat, y_test, s = 1, alpha = 0.05, color = "k")
    ax.set_xlabel("Predicted PWS", fontsize = 18)
    ax.set_ylabel("Actual PWS", fontsize = 18)
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.annotate(f"R$^2$={score:0.2f}", (0.1,0.9),xycoords = "axes fraction", ha = "left")
    return ax

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

def plot_importance_by_category(imp):
    """
    Feature importance combined by categories
    """
    green, brown, blue, yellow, purple, plant, soil, climate, topo, traits \
                                            = get_categories_and_colors()
    combined = pd.DataFrame({"category":["plant","climate","soil","topography","traits"], \
                             "color":[green, blue, brown, yellow, purple]})
    combined = combined.merge(imp.groupby("color").sum(), on = "color")
    
    combined = combined.sort_values("importance")
    fig, ax = plt.subplots(figsize = (3.5,2))
    combined.plot.barh(y = "importance",x="category",color = combined.color, edgecolor = "grey", ax = ax,legend =False )
    # ax.set_yticks(range(len(ticks)))
    # ax.set_yticklabels(ticks)
    
    
    ax.set_xlabel("Variable importance")
    ax.set_ylabel("")
    # ax.set_title("Hydraulic traits' predictive\npower for PWS", weight = "bold")
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
def plot_importance_plants(imp):
    '''
    Feature importance of the plant categories only
    '''
    
    plantsImp = imp[imp['color'] == "yellowgreen"]
    plantsImp = plantsImp.sort_values("importance")
    
    fig, ax = plt.subplots(figsize = (3.5,2))
    plantsImp.plot.barh(y = "importance",x = "symbol", color = plantsImp.color, edgecolor = "grey", ax = ax,legend =False )
    
    ax.set_xlabel("Variable importance")
    ax.set_ylabel("")
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()

def plot_pdp(regr, X_test):
    """
    Partial dependance plot
    requires scikit-learn>=0.24.2
    Parameters
    ----------
    regr : trained rf regression
    X_test : test set data for creating plot
    """
    # Which features need PDPs? print below line and choose the numbers
    # corresponding to the feature
    print(list(zip(X_test.columns, range(X_test.shape[1]))))
    features = np.arange(X_test.shape[1]) #[2,3,7, 12, 4, 13, 11, 18]
    feature_names = list(X_test.columns[features])
    feature_names = prettify_names(feature_names)
    for feature, feature_name in zip(features, feature_names):
        pd_results = sklearn.inspection.partial_dependence(regr, X_test, feature)
        fig, ax = plt.subplots(figsize = (4,4))
        ax.plot(pd_results[1][0], pd_results[0][0])
        ax.set_xlabel(feature_name, fontsize = 18)
        ax.set_ylabel("Plant-water sensitivity", fontsize = 18)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.show()
        
def df_to_raster(dfColumn, rasterShape, lat, lon, geotransform):
    '''
    Take a dataframe and corresponding lat and lon values of same size and turn into raster
    '''

    valMap = np.empty( rasterShape ) * np.nan    
    latInd = np.round( (lat.to_numpy() - geotransform[3])/geotransform[5] ).astype(int)
    lonInd = np.round( (lon.to_numpy() - geotransform[0])/geotransform[1] ).astype(int)
    valMap[latInd, lonInd] = dfColumn

    return valMap 

def plot_map(arrayToPlot, pwsRaster, stateBorders, title, vmin = None, vmax = None, savePath = None):
    '''make map with state borders'''
    
    #preliminary calculatios
    statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']
    pwsExtent = plotting_extent(pwsRaster, pwsRaster.rio.transform())
    
    #actual plotting
    fig, ax = plt.subplots()
    if vmin != None and vmax != None:
        ax = rasterio.plot.show(arrayToPlot, vmin=vmin, vmax=vmax, extent=pwsExtent, ax=ax, cmap='YlOrRd')
    else:
        ax = rasterio.plot.show(arrayToPlot, extent=pwsExtent, ax=ax, cmap='YlOrRd')
    stateBorders[stateBorders['NAME'].isin(statesList)].boundary.plot(ax=ax, edgecolor='black', linewidth=0.5) 
    im = ax.get_images()[0]
    plt.colorbar(im, ax=ax)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    if savePath != None:
        plt.savefig(savePath)
    plt.show()   
    
    
#main code
#%% Load data
path = os.path.join(dirs.dir_data, 'store_plant_soil_topo_climate_PWSthrough2021v3.h5')
df, lat, lon = cleanup_data(path)
df_nopws = df.drop('pws', axis=1)

#%% Train rf
print('running')
start0 = time.time()
X_test, y_test, regrn, score,  imp = regress(df)  
end0 = time.time()

#load from pickle
outfile = open('shapvals.pkl', 'rb')
shapValues = pickle.load(outfile)
outfile.close()
shapNames = 'shap_'+ df_nopws.keys()
shapdf = pd.DataFrame(shapValues, columns=shapNames)

#calculate cross-correlation
corrMat = pd.concat([df_nopws, shapdf], axis=1, keys=['df1', 'df2']).corr().loc['df2', 'df1']
r2bcmap = sns.color_palette("vlag", as_cmap=True)
sns.heatmap(corrMat, 
        xticklabels=prettify_names(df_nopws.columns.values),
        yticklabels=['Shap: ' + s for s in prettify_names(df_nopws.keys())], #'shapdf.columns.values',
        cmap = r2bcmap, vmin=-0.4, vmax=0.4)
plt.savefig('../figures/crossCorrelations.png')

#make a map with state borders
tiffpath = os.path.join("C:/repos/data/pws_features/PWS_through2021.tif") #load an old PWS file. 
statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']
pwsRaster = rxr.open_rasterio(tiffpath, masked=True).squeeze()
pwsExtent = plotting_extent(pwsRaster, pwsRaster.rio.transform())
#put shap in array
ds = gdal.Open(tiffpath)
geotransform = ds.GetGeoTransform()
shapMap = df_to_raster(shapdf['shap_ks'], np.shape(pwsRaster), lat, lon, geotransform)    
statesPath = "C:/repos/data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp"
states = gpd.read_file(statesPath)
plot_map(shapMap, pwsRaster, states, 'shap_ks')

#for each shap column, assign a category
plantShaps, soilShaps, topoShaps, traitsShaps = get_shap_categories()
#for each type, take the sum of values
dfPlant = shapdf[plantShaps]
groupShapPlant = np.sum(np.abs(shapdf[plantShaps]), axis=1)
groupShapSoil = np.sum(np.abs(shapdf[soilShaps]), axis=1)
groupShapTopo = np.sum(np.abs(shapdf[topoShaps]), axis=1)
groupShapTraits = np.sum(np.abs(shapdf[traitsShaps]), axis=1)

#plot in a subplot file, each with same colorbars
plantMap = df_to_raster(groupShapPlant, np.shape(pwsRaster), lat, lon, geotransform)
plot_map(plantMap, pwsRaster, states, 'sum of veg density Shapley values', vmin=0, vmax=0.6, savePath='../figures/plantTotShap.png')
soilMap = df_to_raster(groupShapSoil, np.shape(pwsRaster), lat, lon, geotransform)
plot_map(soilMap, pwsRaster, states, 'sum of soil Shapley values', vmin=0, vmax=0.6, savePath='../figures/soilTotShap.png')
topoMap = df_to_raster(groupShapTopo, np.shape(pwsRaster), lat, lon, geotransform)
plot_map(topoMap, pwsRaster, states, 'sum of topo Shapley values', vmin=0, vmax=0.6, savePath='../figures/topoTotShap.png')
traitsMap = df_to_raster(groupShapTraits, np.shape(pwsRaster), lat, lon, geotransform)
plot_map(traitsMap, pwsRaster, states, 'sum of trait Shapley values', vmin=0, vmax=0.6, savePath='../figures/traitsTotShap.png')

#map all individual features
for feat in df_nopws.columns:
    figFile = '../figures/' + feat + '.png'
    featMap = df_to_raster(df_nopws[feat], np.shape(pwsRaster), lat, lon, geotransform)
    plot_map(featMap, pwsRaster, states, prettify_names([feat])[0], savePath = figFile)

'''to add: 
    adjust vmin, vmax on feat maps
 PFT colormap. pft has values 41, 42, 43, 52, 71, 81
 '''