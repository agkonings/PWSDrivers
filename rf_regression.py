# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from osgeo import gdal
import sklearn.ensemble
import sklearn.model_selection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import mpl_scatter_density
import sklearn.preprocessing
import matplotlib.patches
import sklearn.inspection
from sklearn.inspection import permutation_importance
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
import pickle
import dill
import dirs
from alepython import ale_plot
import geopandas as gpd
from shapely.geometry import Point
import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent, show

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
    
    plant = ["canopy_height", "agb",'ndvi', "nlcd","species"]
    soil = ['thetas', 'ks','Sr','Sbedrock','bulk_density','theta_third_bar','AWS']
    climate = ['vpd_mean', 'vpd_std',"ppt_mean","ppt_cv","t_mean","t_std","AI"]
    topo = ['aspect', 'slope', 'twi','HAND']
    
    return green, brown, blue, yellow, plant, soil, climate, topo 

def prettify_names(names):
    new_names = {"ks":"K$_{s,max}$",
                 "ndvi":"$NDVI_{mean}$",
                 "vpd_mean":"VPD$_{mean}$",
                 "vpd_cv":"VPD$_{CV}$",
                 "vpd_std":"VPD$_{std}$",
                 "ppt_mean":"Precip$_{mean}$",
                 "ppt_cv":"Precip$_{CV}$",
                 "agb":"Biomass",
                 "canopy_height": "Canopy height",
                 "root_depth":"Root depth",
                 "bulk_density":"Bulk density",
                 "nlcd": "Land cover",                
                 "aspect":"Aspect",
                 "slope":"Slope",
                 "twi":"TWI",
                 "t_mean":"Temp$_{mean}$",
                 "t_std":"Temp$_{st dev}$",
                 "lon":"Lon", "lat":"Lat",
                 "theta_third_bar": "$\psi_{0.3}$",
                 "AWS":"Avail water storage",
                 "AI":"Aridity index",
                 "Sr": "RZ storage",
                 "species":"species",
                 "HAND":"HAND"
                 }
    return [new_names[key] for key in names]

def prettify_names_wunits(names):
    new_names = {"ks":"K$_{s,max}$ [$\mu$m/s]",
                 "ndvi":"$NDVI_{mean} [-]$",
                 "vpd_mean":"VPD$_{mean}$ [hPa]",
                 "ppt_cv":"Precip$_{CV}$ [-]",
                 "monsoon_index":"Monsoon index [-]",
                 "bulk_density":"Bulk density [g/cm$^3$]",                
                 "aspect":"Aspect [$^o$]",
                 "slope":"Slope [%]",
                 "twi":"TWI [-]",
                 "AI":"Aridity index [-]",
                 "Sr": "RZ storage [mm]"}
    return [new_names[key] for key in names]
    
    
def regress(df, optHyperparam=False):
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
    
    '''
    # Checking if leaves or node_impurity affects performance
    # after running found that it has almost no effect (R2 varies by 0.01)
    '''
    if optHyperparam is True:
        # separate into train and test set
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1, random_state=32) 
        for leaves in [3, 4, 15]: #[6,7,8,9,10,12, 14, 15]:
            for decrease in [ 1e-8, 1e-10]: 
                for nEst in [50,120,200,500,600]: #[50,90,120,140]: 
                        for depth in [8, 15]: #8, 15, 25
                            for max_feat in ['sqrt','auto']:
                                # construct rf model
                                regrn = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                                    max_depth = depth, min_impurity_decrease=decrease, n_estimators = nEst, max_features=max_feat)
                                # train
                                regrn.fit(X_train, y_train)
                                # test set performance
                                scoreTrain = regrn.score(X_train, y_train)
                                score = regrn.score(X_test,y_test)             
                                print(f"[INFO] scoreTrain={scoreTrain:0.3f}, score={score:0.3f}, leaves={leaves}, decrease={decrease}, nEst = {nEst}, depth={depth}, nSplit={max_feat}")                                
                            
                           
    #can get highest with 3 leaves, 120 nEst, decrease 1e-8, but that seems like low number of leaves
    #old configuration was leaves = 6, decrease 1e-6, nEst = 50
    leaves = 4
    decrease = 1e-8
    depth = 8
    nEst = 120
    max_feat = 'auto'
    # construct rf model

    #naive CV does horribly because KFold splits along indices, but the data are 
    #geographically distributed. So use a shuffled dataset to get better folds
    shuffled_df = df.sample(frac=1)
    shuffledy = shuffled_df['pws']
    shuffledX = shuffled_df.drop("pws", axis = 1)    
    
    #core cross-validation calcluations
    folds = sklearn.model_selection.KFold(n_splits=10)
    scoresCV = []
    trainIndStore = []
    testIndStore = []
    for train, test in folds.split(shuffledX):
        trainIndStore.append(train)
        testIndStore.append(test)
        X_train_CV, X_test_CV, y_train_CV, y_test_CV = shuffledX.iloc[train], shuffledX.iloc[test], shuffledy.iloc[train], shuffledy.iloc[test]
        regrnCV = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                                    max_depth = depth, min_impurity_decrease=decrease, n_estimators = nEst, max_features='auto')
        regrnCV.fit(X_train_CV, y_train_CV)
        scoresCV.append(regrnCV.score(X_test_CV, y_test_CV))
    print(f"[CV Score stats] mean={np.mean(scoresCV):0.3f}, std={np.std(scoresCV):0.3f}")
    print("full CV scores:", str(scoresCV))
    
    #re-construct final RF model, from fold with performance closest to mean
    #foo, meanFold = np.min( np.abs(scoresCV - np.mean(scoresCV)) )
    meanFold = np.argmin( np.abs(scoresCV - np.mean(scoresCV)) )
    X_train, X_test = shuffledX.iloc[trainIndStore[meanFold]], shuffledX.iloc[testIndStore[meanFold]]
    y_train, y_test = shuffledy.iloc[trainIndStore[meanFold]], shuffledy.iloc[testIndStore[meanFold]]
        
    # train
    regrn = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, max_depth=depth, \
                  min_impurity_decrease=decrease, n_estimators = nEst, max_features='auto')
    regrn.fit(X_train, y_train)
    # test set performance
    score = regrn.score(X_test,y_test)
    scoreTrain = regrn.score(X_train, y_train)
    print(f"[INFOTrain] score={scoreTrain:0.3f}, leaves={leaves}, decrease={decrease}")
    print(f"[INFO] score={score:0.3f}, leaves={leaves}, decrease={decrease}")
    
    # assemble all importance with feature names and colors
    #rImp = permutation_importance(regrn, X_test, y_test,
    #                        n_repeats=2, random_state=0)
    rImp = permutation_importance(regrn, X_test, y_test,
                            n_repeats=2, random_state=8)
    heights = rImp.importances_mean
    uncBars = rImp.importances_std
    #heights = regrn.feature_importances_
    ticks = X.columns
  
    green, brown, blue, yellow, plant, soil, climate, topo, \
                                            = get_categories_and_colors()
                                            

    
    imp = pd.DataFrame(index = ticks, columns = ["importance"], data = heights)
    imp['importance std'] = uncBars
    
    def _colorize(x):
        if x in plant:
            return green
        elif x in soil:
            return brown
        elif x in climate:
            return blue
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
    ax.scatter(y_hat, y_test, s = 1, alpha = 0.05, color='k')
    ax.set_xlabel("Predicted PWS", fontsize = 18)
    ax.set_ylabel("Actual PWS", fontsize = 18)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.annotate(f"R$^2$={score:0.2f}", (0.1,0.9),xycoords = "axes fraction", ha = "left")
    return ax

def plot_error_pattern(path, df):
    """
    Make map of prediction error to visually test if there is a spatial pattern
    Also plot other inputs for comparison

    Parameters
    ----------
    path: location where H5 file with PWS and all input features is stored
    df: dataframe with features

    Returns
    -------
    ax: axis handle

    """
    
    #make map_predictionError function later
    X_test, y_test, regrn, score,  imp = regress(df)
    
    XAll = df.drop("pws",axis = 1)
    y_hat = regrn.predict(XAll)
    predError = y_hat - df['pws']
    
    filename = os.path.join("C:/repos/data/pws_features/PWS_through2021.tif") #load an old PWS file. 
    ds = gdal.Open(filename)
    geotransform = ds.GetGeoTransform()
    pws = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    errorMap = np.empty( np.shape(pws) ) * np.nan
    
    store = pd.HDFStore(path)
    df2 =  store['df']   # save it
    store.close()
    df2.dropna(inplace = True)
    
    latInd = np.round( (df2['lat'].to_numpy() - geotransform[3])/geotransform[5] ).astype(int)
    lonInd = np.round( (df2['lon'].to_numpy() - geotransform[0])/geotransform[1] ).astype(int)
    errorMap[latInd, lonInd] = predError
    
    
    fig, ax1 = plt.subplots()
    im = ax1.imshow(errorMap, interpolation='none', 
                   vmin=1, vmax=1.5)
    plt.title('prediction error')
    cbar = plt.colorbar(im)

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
    
    fig = plt.subplots(2, 1, figsize = (4,10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax0 = plt.subplot(gs[0])
    green, brown, blue, yellow, plant, soil, climate, topo \
                                            = get_categories_and_colors()
    
    
    imp.plot.barh(y = "importance",x="symbol",color = imp.color, edgecolor = "grey", ax = ax0, fontsize = 18)

    '''
    legend_elements = [matplotlib.patches.Patch(facecolor=blue, edgecolor='grey',
                             label='Climate'),
                       matplotlib.patches.Patch(facecolor=green, edgecolor='grey',
                             label='Veg density'),         
                       matplotlib.patches.Patch(facecolor=yellow, edgecolor='grey',
                             label='Topography'), 
                       matplotlib.patches.Patch(facecolor=brown, edgecolor='grey',
                             label='Soil')]
    '''
    #ax0.legend(handles=legend_elements, fontsize = 18, loc='lower right')
    ax0.set_xlabel("Variable importance", fontsize = 18)
    ax0.set_ylabel("")
    ax0.set_xlim(0,0.50)

    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.get_legend().remove()
    ax0.text(-0.5, 1.05, 'a)', transform=ax0.transAxes, fontsize = 18, fontweight='bold', va='top', ha='left')
    
    #then make second panel with importance by categories    
    combined = pd.DataFrame({"category":["Veg density","Climate","Soil","Topography"], \
                             "color":[green, blue, brown, yellow]})
    combined = combined.merge(imp.groupby("color").sum(), on = "color")
    
    combined = combined.sort_values("importance")
    
    ax1 = plt.subplot(gs[1])
    combined.plot.barh(y = "importance",x="category",color = combined.color, edgecolor = "grey", ax = ax1,fontsize = 18, legend =False )
    ax1.set_xlabel("Variable importance", fontsize = 18)
    ax1.set_ylabel("")
    ax1.set_xlim(0,0.50)
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.text(-0.5, 1.05, 'b)', transform=ax1.transAxes, fontsize = 18, fontweight='bold', va='top', ha='left')
     
    #plt.savefig("../figures/PWSDriversPaper/importanceCombined.jpeg", dpi=300, bbox_inches='tight')
    plt.show()
    
    
    return plt

def plot_importance_by_category(imp):
    """
    Feature importance combined by categories
    """

    
    green, brown, blue, yellow, plant, soil, climate, topo \
                                            = get_categories_and_colors()
    combined = pd.DataFrame({"category":["Veg density","Climate","Soil","Topography"], \
                             "color":[green, blue, brown, yellow]})
    combined = combined.merge(imp.groupby("color").sum(), on = "color")
    
    combined = combined.sort_values("importance")
    fig, ax = plt.subplots(figsize = (3.5,2))
    combined.plot.barh(y = "importance",x="category",color = combined.color, edgecolor = "grey", ax = ax,legend =False )
    # ax.set_yticks(range(len(ticks)))
    # ax.set_yticklabels(ticks)
    ax1.set_xlabel("Variable importance")
    ax1.set_ylabel("")
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
     
    plt.tight_layout()
    plot.show()
        
    return plt

def plot_top_ale(regr, X_test, savePath = None):
    """
    Accumulated local effects plot of top features 
    Manually limit to top features for simplicity of coding
    Parameters
    ----------
    regr : trained rf regression
    X_test : test set data for creating plot
    
    """
    features = ['ndvi','vpd_mean', 'slope','ppt_cv']
    feature_names = prettify_names_wunits(features)
    ftCnt = 0
    figALEs, axs = plt.subplots(nrows=2, ncols=2, figsize = (6,6))
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    for feature, feature_name, ax in zip(features, feature_names, axs.ravel()):
        axALE = ale_plot(regr, X_test, feature, monte_carlo=False, rugplot_lim=None)  
        #pull out data and re-plot
        lines = axALE.lines
        for line in lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
        plt.figure(figALEs.number)    
        ax.plot(x_data, y_data, color="black")
        ax.set_xlabel(feature_name, fontsize = 18)
        if feature is 'ndvi' or feature is 'slope': #lable only left column
            ax.set_ylabel('ALE', fontsize = 18)
        if feature is 'vpd_mean':
            ax.set_xlim(-0.1,35) #cut off very extreme range where have just a few distribution points
        if feature is 'slope':
            ax.set_xlim(-0.1,10.8) #cut off very extreme range where have just a few distribution points
            ax.xaxis.set_ticks(np.arange(3, 10, 3))
        #if feature is 'monsoon_index':
        #    ax.xaxis.set_ticks(np.arange(0, 0.7, 0.2))            
        ax.tick_params(axis='both', labelsize = 14)
        sns.rugplot(X_test[feature], ax=ax, alpha=0.2)    

    
    figALEs.show()
    if savePath != None:
        figALEs.savefig(savePath, dpi=600, bbox_inches='tight')
    return plt


def plot_R2_by_category(singleCat):
    """
    Feature importance combined by categories
    """
    
    singleCat = singleCat.sort_values("score", ascending=False)
    
    fig, ax = plt.subplots(figsize = (2, 3.5)) 
    
    singleCat.plot.bar(y = "score",x="labels", color = singleCat.colors, edgecolor = "grey", ax = ax,legend =False )
    ax.set_xlabel("Variable importance")
    ax.set_ylabel("R$^2$")  
    ax.set_xlabel("")  
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
     
    return plt

def regress_per_category(df, optHyperparam=False):
    """
    Build a dataframe with the scores for a model built on each category only
    """
    
    #Run a few alternative versions of the RF model with reduced inputs
    print('only climate')
    df_onlyClimate = df.copy()
    df_onlyClimate = df_onlyClimate[['pws','vpd_mean','ppt_cv','AI']]
    X_test_oC, y_test_oC, regrn_oC, score_onlyClimate, imp_oC = regress(df_onlyClimate, optHyperparam=False)
    print('only NDVI')
    df_onlyPlant = df.copy()
    df_onlyPlant = df_onlyPlant[['pws','ndvi']]
    X_test_oP, y_test_oP, regrn_oP, score_onlyPlant, imp_oP = regress(df_onlyPlant, optHyperparam=False)
    print('only Soil')
    df_onlySoil = df.copy()
    df_onlySoil = df_onlySoil[['pws','ks','bulk_density','Sr']]
    X_test_oS, y_test_oS, regrn_oS, score_onlySoil, imp_oS = regress(df_onlySoil, optHyperparam=False)
    print('only topo')
    df_onlyTopo = df.copy()
    df_onlyTopo = df_onlyTopo[['pws','aspect','slope','twi']]
    X_test_oT, y_test_oT, regrn_oT, score_onlyTopo, imp_oT = regress(df_onlyTopo, optHyperparam=False)
    
    green, brown, blue, yellow, plant, soil, climate, topo \
                                            = get_categories_and_colors()
    cats = ['Climate','Veg density','Topography','Soil']
    scores = [score_onlyClimate, score_onlyPlant, score_onlyTopo, score_onlySoil]
    colors = [blue, green, brown, yellow]
    data = {'score': scores, 'colors': colors, 'labels': ['Only climate','Only NDVI','Only topography','Only soil']}
    singleCat = pd.DataFrame(data)

    return singleCat

def plot_map(arrayToPlot, pwsExtent, stateBorders, title = None, vmin = None, vmax = None, clrmap = 'YlGnBu', savePath = None):
    '''make map with state borders'''
    
    #preliminary calculatios
    statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']    
    
    #actual plotting
    fig, ax = plt.subplots()
    if vmin != None and vmax != None:
        ax = rasterio.plot.show(arrayToPlot, interpolation='nearest', vmin=vmin, vmax=vmax, extent=pwsExtent, ax=ax, cmap=clrmap)
    else:
        ax = rasterio.plot.show(arrayToPlot, interpolation='nearest', extent=pwsExtent, ax=ax, cmap=clrmap)
    stateBorders[stateBorders['NAME'].isin(statesList)].boundary.plot(ax=ax, edgecolor='black', linewidth=0.5) 
    im = ax.get_images()[0]
    #cbar = plt.colorbar(im, ax=ax) #ticks=range(0,6)
    #cbar.ax.set_xticklabels([ 'Deciduous','Evergreen','Mixed','Shrub','Grass', 'Pasture'])
    plt.title(title)
    ax.axis('off')
    plt.xticks([])
    plt.yticks([])
    if savePath != None:
        plt.savefig(savePath)
    plt.show() 
    
plt.rcParams.update({'font.size': 18})

#%% Load data
dfPath = os.path.join(dirs.dir_data, 'inputFeatures_wgNATSGO_wBA_wHAND.h5')
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_allSeas_nonorm_4monthslag_exact6years.tif'
df_wSpec =  load_data(dfPath, pwsPath)

#sdroppedvarslist based on manual inspection so no cross-correlations greater than 0.75, see pickFeatures.py
#further added nlcd to drop list since doesn't really make sense if focusing on fia plots
droppedVarsList = ['elevation','dry_season_length','t_mean','ppt_mean','t_cv','ppt_lte_100',
                'canopy_height', 'HAND','restrictive_depth','clay',
                'dist_to_water','basal_area','theta_third_bar','AWS','sand',
                'agb','p50','gpmax','vpd_cv','root_depth','g1','c','isohydricity']
df_wSpec = cleanup_data(df_wSpec, droppedVarsList)

#remove pixels with NLCD status that is not woody
df_wSpec = df_wSpec[df_wSpec['nlcd']<70] #unique values are 41, 42, 43, 52
#remove mixed forest
df_wSpec = df_wSpec[df_wSpec['nlcd'] != 43] #unique values are 41, 42, 43, 52

'''
#save for exact use in checkCrossCorrs.py
pickleLoc = 'C:/repos/pws_drivers/data/df_wSpec.pkl'
with open(pickleLoc, 'wb') as file:
    pickle.dump(df_wSpec, file)
'''
    
#then drop species, lat, lon for actual RF
df_noSpec = df_wSpec.drop(columns=['lat','lon','species', 'nlcd'], inplace=False)

#seems to be some weird issue where RF model and importance is possibly affected by number of unique pixels in each dataset
#add small random noise to avoid that to be safe
uniqueCnt = df_noSpec.nunique()
for var in df_noSpec.columns:
    if uniqueCnt[var] < 10000:
        reasonableNoise = 1e-5*df_noSpec[var].median()
        df_noSpec[var] = df_noSpec[var] + np.random.normal(0, reasonableNoise, len(df_noSpec))


#now actually train model on everything except the species
#Replace trained model with pickled version
#prevMod = dill.load( open('C:/repos/data/RFregression_dill_backup_beforeRound2.pkl', 'rb') )
prevMod = dill.load( open('C:/repos/data/rf_regression_dill.pkl', 'rb') )
regrn = getattr(prevMod, 'regrn')
score = getattr(prevMod, 'score')
imp = getattr(prevMod, 'imp')
X_test = getattr(prevMod, 'X_test')
y_test = getattr(prevMod, 'y_test')
'''
# old code:
# Train rf#
X_test, y_test, regrn, score,  imp = regress(df_noSpec, optHyperparam=False)  
'''

# make plots
ax = plot_corr_feats(df_noSpec)
pltImp = plot_importance(imp)
#pltALE = plot_top_ale(regrn, X_test, savePath = "../figures/PWSDriversPaper/ales.jpeg")
pltALE = plot_top_ale(regrn, X_test, savePath = None)


'''
now check how explanatory power compares if don't have species 
'''
print('now doing species power calculations')    
print('predictive ability with species alone')
print( 'number of pixels studied: ' + str(len(df_wSpec)) ) 
pwsVec = df_wSpec['pws']
specVec = df_wSpec['species']
pwsPred = np.zeros(pwsVec.shape)
specCount = df_wSpec['species'].value_counts()
minFreq = 5
for specCode in np.unique(df_wSpec['species']):
    if specCount[specCode] > minFreq:
        #differentiating (obs_i-X)^2 shows that optimal predictor is mean of each cat
        thisMean = np.mean( pwsVec[specVec == specCode] )
        pwsPred[specVec == specCode] = thisMean
        

#next line is hack to make sure species with less than minFreq occurences don't count
pwsVec[pwsPred == 0] = 0
resPred = pwsVec - pwsPred
SSres = np.sum(resPred**2)
SStot = np.sum( (pwsVec - np.mean(pwsVec))**2 ) #total sum of squares
coeffDeterm = 1 - SSres/SStot

print('amount explained with ONLY species info ' + str(coeffDeterm))
print('fraction explained by species' + str(coeffDeterm/score))

'''Plot Figure 3 with R2 for both RF and species'''
y_hat =regrn.predict(X_test)
#get full series of all predictions on all points (not separate train and test splits)
rfPredAll =regrn.predict(df_noSpec.drop("pws",axis = 1))

xySp = np.vstack([pwsPred,pwsVec])
kdeSp = gaussian_kde(xySp, bw_method=0.05)(xySp)
xyRF = np.vstack([rfPredAll,pwsVec])
kdeRF = gaussian_kde(xyRF, bw_method=0.05)(xyRF)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(pwsPred, pwsVec, c=kdeSp, s = 1, alpha = 0.4, cmap='inferno')
ax1.set_box_aspect(1)
ax1.set_xlabel("Predicted PWS", fontsize = 14)
ax1.set_ylabel("Actual PWS", fontsize = 14)
ax1.xaxis.set_ticks(np.arange(0, 6.2, 1))
ax1.yaxis.set_ticks(np.arange(0, 6.2, 1))
ax1.plot([0, 6], [0, 6], color='b', linestyle='--')
ax1.set_xlim(0,6), ax1.set_ylim(0,6)
ax1.set_title('Species mean', fontsize = 14)
ax1.annotate(f"R$^2$={coeffDeterm:0.2f}", (0.61,0.06),xycoords = "axes fraction", 
             fontsize=14, ha = "left")
ax1.annotate('a)', (-0.2,1.1),xycoords = "axes fraction", 
             fontsize=14, weight='bold')
ax2.set_box_aspect(1)
ax2.scatter(rfPredAll, pwsVec, c=kdeRF, s = 1, alpha = 0.4, cmap='inferno')
ax2.set_xlabel("Predicted PWS", fontsize = 14)
ax2.set_xlim(0,6), ax2.set_ylim(0,6)
ax2.xaxis.set_ticks(np.arange(0, 6.2, 1))
ax2.yaxis.set_ticks(np.arange(0, 6.2, 1))
ax2.plot([0, 6], [0, 6], color='b', linestyle='--', )
ax2.set_title('Random forest', fontsize = 14)
ax2.annotate(f"R$^2$={score:0.2f}", (0.61,0.06),xycoords = "axes fraction", 
             fontsize=14, ha = "left")
ax2.annotate('b)', (-0.2,1.10),xycoords = "axes fraction", 
             fontsize=14, weight='bold')
fig.tight_layout()
plt.savefig("G:/My Drive/0000WorkComputer/dataStanford/PWSpeciesAnalysis/reconstructedv4densityPlotsModels.png", dpi=600)

''' 
Make some maps of PWS as observed, predicted, and error
'''
#make dataframes
dfPWS = df_wSpec[['lat','lon','pws']]
dfSpPred = df_wSpec[['lat','lon']].copy()
dfSpPred['pws'] = pwsPred
dfRFPred = df_wSpec[['lat','lon']].copy()
dfRFPred['pws'] = rfPredAll

#prep plotting stuff
statesPath = "C:/repos/data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp"
states = gpd.read_file(statesPath)    
stateBorders = states.to_crs(epsg=5070)
statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']    
stateBorders = stateBorders[stateBorders['NAME'].isin(statesList)]    



#Plot PWS 
fig = plt.figure(figsize=(10, 3))
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.2])
ax1 = fig.add_subplot(gs[0])
# Convert the dataframe to a GeoDataFrame
geometry = [Point(xy) for xy in zip(dfPWS['lon'], dfPWS['lat'])]
gdf = gpd.GeoDataFrame(dfPWS, geometry=geometry)
# Set the coordinate reference system to WGS84, then project to Albers (5070)
gdf.set_crs(epsg=4326, inplace=True)
gdf_albers = gdf.to_crs(epsg=5070)
stateBorders.boundary.plot(ax=ax1, linewidth=1, color='black')
ax1 = gdf_albers.plot(column='pws', cmap='cool', vmin=0, vmax=5, markersize=0.5, ax=ax1)
ax1.set_title('True PWS', fontsize=18)
ax1.axis('off')
plt.xticks([])
plt.yticks([])
ax2 = fig.add_subplot(gs[1])
# Convert the dataframe to a GeoDataFrame
geometry = [Point(xy) for xy in zip(dfSpPred['lon'], dfSpPred['lat'])]
gdf = gpd.GeoDataFrame(dfSpPred, geometry=geometry)
# Set the coordinate reference system to WGS84, then project to Albers (5070)
gdf.set_crs(epsg=4326, inplace=True)
gdf_albers = gdf.to_crs(epsg=5070)
stateBorders.boundary.plot(ax=ax2, linewidth=1, color='black')
ax2 = gdf_albers.plot(column='pws', cmap='cool', vmin=0, vmax=5, markersize=0.5, ax=ax2)
ax2.set_title('Species PWS', fontsize=18)
ax2.axis('off')
plt.xticks([])
plt.yticks([])
ax3 = fig.add_subplot(gs[2])
# Convert the dataframe to a GeoDataFrame
geometry = [Point(xy) for xy in zip(dfRFPred['lon'], dfRFPred['lat'])]
gdf = gpd.GeoDataFrame(dfRFPred, geometry=geometry)
# Set the coordinate reference system to WGS84, then project to Albers (5070)
gdf.set_crs(epsg=4326, inplace=True)
gdf_albers = gdf.to_crs(epsg=5070)
stateBorders.boundary.plot(ax=ax3, linewidth=1, color='black')
ax3 = gdf_albers.plot(column='pws', cmap='cool', vmin=0, vmax=5, markersize=0.5, ax=ax3)
ax3.set_title('PWS from RF', fontsize=18)
ax3.axis('off')
plt.xticks([])
plt.yticks([])
#add same colorbar for all
cbar_ax = fig.add_subplot(gs[3])
sm = plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(vmin=0, vmax=5))
sm._A = []
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=16)
#plt.savefig("../figures/PWSDriversPaper/predictedPWSMaps.jpeg", dpi=300)

'''
For reviewer 1/suppmat, calculate model performance without NDVI or VPD
'''
singleCat = regress_per_category(df_noSpec)
pltR2Cat = plot_R2_by_category(singleCat)
#pltR2Cat.savefig("../figures/PWSDriversPaper/R2OnlyCategories.jpeg", dpi=300)
print('no VPD')
df_noVPD = df_noSpec.copy()
df_noVPD.drop('vpd_mean', axis = 1, inplace = True)
X_test_nV, y_test_nV, regrn_nV, score_noVPD, imp_nV = regress(df_noVPD, optHyperparam=False)
print('no NDVI')
df_noNDVI = df_noSpec.copy()
df_noNDVI.drop('ndvi', axis = 1, inplace = True)
X_test_nN, y_test_nN, regrn_nN, score_noNDVI, imp_nN = regress(df_noNDVI, optHyperparam=False)
print('no VPD and no NDVI')
df_noVPDnoNDVI = df_noSpec.copy()
df_noVPDnoNDVI.drop('ndvi', axis = 1, inplace = True)
df_noVPDnoNDVI.drop('vpd_mean', axis = 1, inplace = True)
X_test_nVnN, y_test_nVnN, regrn_nVnN, score_noVPDnoNDVI, imp_nVnN = regress(df_noVPDnoNDVI, optHyperparam=False)


#dill.dump_session('C:/repos/data/rf_regression_dill.pkl')
print('saved ok')


