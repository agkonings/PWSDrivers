# -*- coding: utf-8 -*-
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
import dill
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
    
    plant = ['basal_area','canopy_height', "agb",'ndvi', "nlcd","species",'isohydricity', 'root_depth', 'p50', 'gpmax', 'c', 'g1']
    soil = ['clay', 'sand','silt','thetas', 'ks', 'vanGen_n','Sr','Sbedrock','bulk_density','theta_third_bar','AWS']
    climate = ['vpd_mean', 'vpd_std',"ppt_mean","ppt_cv","t_mean","t_std","ppt_lte_100","AI"]
    topo = ['elevation', 'aspect', 'slope', 'twi',"dist_to_water"]
    
    return green, brown, blue, yellow, plant, soil, climate, topo 

def prettify_names(names):
    new_names = {"ks":"K$_{s,max}$",
                 "ndvi":"$NDVI_{mean}$",
                 "vpd_mean":"VPD$_{mean}$",
                 "vpd_cv":"VPD$_{CV}$",
                 "vpd_std":"VPD$_{std}$",
                 "thetas":"Soil porosity",
                 "elevation":"Elevation",
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
                 "dist_to_water":"Dist to water",
                 "t_mean":"Temp$_{mean}$",
                 "t_std":"Temp$_{st dev}$",
                 "lon":"Lon", "lat":"Lat",
                 "theta_third_bar": "$\psi_{0.3}$",
                 "vanGen_n":"van Genuchten n",
                 "AWS":"Avail water storage",
                 "AI":"Aridity index",
                 "Sr": "RZ storage",
                 "restrictive_depth": "Restricton depth",
                 "species":"species",
                 "basal_area": "Basal area",
                 "HAND":"HAND"
                 }
    return [new_names[key] for key in names]

def prettify_names_wunits(names):
    new_names = {"ks":"K$_{s,max}$ [$\mu$m/s]",
                 "ndvi":"$NDVI_{mean} [-]$",
                 "vpd_mean":"VPD$_{mean}$ [hPa]",
                 "ppt_cv":"Precip$_{CV}$ [-]",
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
    # separate into train and test set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1, random_state=32) 
    
    '''
    # Checking if leaves or node_impurity affects performance
    # after running found that it has almost no effect (R2 varies by 0.01)
    '''
    if optHyperparam is True:
        for leaves in [3, 8, 15]: #[6,7,8,9,10,12, 14, 15]:
            for decrease in [ 1e-8, 1e-10]: 
                for nEst in [50,120]: #[50,90,120,140]: 
                        for depth in [8, 15]: #8, 15, 25
                            # construct rf model
                            regrn = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                                  max_depth = depth, min_impurity_decrease=decrease, n_estimators = nEst)
                            # train
                            regrn.fit(X_train, y_train)
                            # test set performance
                            score = regrn.score(X_test,y_test)             
                            print(f"[INFO] score={score:0.3f}, leaves={leaves}, decrease={decrease}, nEst = {nEst}, depth={depth}")
                            # choose min leaves in terminal node and node impurity
                            
                           
    #can get highest with 3 leaves, 120 nEst, decrease 1e-8, but that seems like low number of leaves
    #old configuration was leaves = 6, decrease 1e-6, nEst = 50
    leaves = 4
    decrease = 1e-8
    depth = 8
    nEst = 120
    # construct rf model
    regrn = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, max_depth=depth, \
                      min_impurity_decrease=decrease, n_estimators = nEst)
    # train
    regrn.fit(X_train, y_train)
    # test set performance
    score = regrn.score(X_test,y_test)
    scoreTrain = regrn.score(X_train, y_train)
    #print(f"[INFOTrain] score={scoreTrain:0.3f}, leaves={leaves}, decrease={decrease}")
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
#    # Load data
#    path = os.path.join(dirs.dir_data, 'store_plant_soil_topo_climate_PWSthrough2021v2.h5')
#    df = cleanup_data(path)
    
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
    
    fig, ax = plt.subplots(figsize = (5.5,7))
    green, brown, blue, yellow, plant, soil, climate, topo \
                                            = get_categories_and_colors()

    imp.plot.barh(y = "importance",x="symbol",color = imp.color, edgecolor = "grey", ax = ax, fontsize = 18)

    legend_elements = [matplotlib.patches.Patch(facecolor=blue, edgecolor='grey',
                             label='Climate'),
                       matplotlib.patches.Patch(facecolor=green, edgecolor='grey',
                             label='Veg density'),         
                       matplotlib.patches.Patch(facecolor=yellow, edgecolor='grey',
                             label='Topography'), 
                       matplotlib.patches.Patch(facecolor=brown, edgecolor='grey',
                             label='Soil')]
    ax.legend(handles=legend_elements, fontsize = 18, loc='lower right')
    ax.set_xlabel("Variable importance", fontsize = 18)
    ax.set_ylabel("")
    ax.set_xlim(0,0.50)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    return plt

def plot_importance_by_category(imp, dropVars = None):
    """
    Feature importance combined by categories
    """
    
    if dropVars  != None:
        imp = imp.drop(dropVars)
    
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
    
    
    ax.set_xlabel("Variable importance")
    ax.set_ylabel("")
    # ax.set_title("Hydraulic traits' predictive\npower for PWS", weight = "bold")
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    print(combined)
    return plt
    
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
    feature_names = prettify_names_wunits(feature_names)
    ftCnt = 0
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize = (12,9))
    plt.subplots_adjust(hspace=0.7)
    plt.subplots_adjust(wspace=0.5)
    for feature, feature_name, ax in zip(features, feature_names, axs.ravel()):
        pd_results = sklearn.inspection.partial_dependence(regr, X_test, feature)       
        #fig, ax = plt.subplots(figsize = (4,4))        
        #plt.subplot(5,2,ftCnt)
        ax.plot(pd_results[1][0], pd_results[0][0], color="black")
        ax.set_xlabel(feature_name, fontsize = 18)
        if np.mod(ftCnt, 3) == 0:
            ax.set_ylabel("PWS", fontsize = 18)
        #axs[rowCnt, colCnt]
        ax.tick_params(axis='both', labelsize = 16)
        ftCnt += 1
    
    fig.delaxes(axs[3][1])
    fig.delaxes(axs[3][2])
    return plt

def plot_scatter_feats(df_noSpec):
    """
    Plot joint KDE across features to get a feel for whether residual
    co-variates lurking in certain sub-areas
    """
    g = sns.PairGrid(df_noSpec)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    
    return g
    
    
plt.rcParams.update({'font.size': 18})


#%% Load data
dfPath = os.path.join(dirs.dir_data, 'inputFeatures_wgNATSGO_wBA_wHAND.h5')
pwsPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_allSeas_nonorm_4monthslag_exact6years.tif'
df_wSpec =  load_data(dfPath, pwsPath)

#sdroppedvarslist based on manual inspection so no cross-correlations greater than 0.75, see pickFeatures.py
#further added nlcd to drop list since doesn't really make sense if focusing on fia plots
droppedVarsList = ['elevation','dry_season_length','t_mean','ppt_mean','t_cv','ppt_lte_100',
                'canopy_height', 'HAND','restrictive_depth','clay',
                'dist_to_water','basal_area','theta_third_bar', 'AWS','sand',
                'agb','p50','gpmax','vpd_cv','root_depth']
droppedVarsList = droppedVarsList + ['g1','c','isohydricity']
df_wSpec = cleanup_data(df_wSpec, droppedVarsList)

#remove pixels with NLCD status that is not woody
df_wSpec = df_wSpec[df_wSpec['nlcd']<70] #unique values are 41, 42, 43, 52
#remove mixed forest
df_wSpec = df_wSpec[df_wSpec['nlcd'] != 43] #unique values are 41, 42, 43, 52

'''
#save for exact use in checkCrossCorrs.py
pickleLoc = '../data/df_wSpec.pkl'
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

'''
now actually train model on everything except the species
'''
#Replace trained model with pickled version
prevMod = dill.load( open('./RFregression_dill.pkl', 'rb') )
regrn = getattr(prevMod, 'regrn')
score = getattr(prevMod, 'score')
imp = getattr(prevMod, 'imp')
X_test = getattr(prevMod, 'X_test')
y_test = getattr(prevMod, 'y_test')
'''
# old code:
# Train rf
X_test, y_test, regrn, score,  imp = regress(df_noSpec, optHyperparam=False)  
'''
# make plots
ax = plot_corr_feats(df_noSpec)
pltImp = plot_importance(imp)
pltImp.savefig("../figures/PWSDriversPaper/importance.jpeg", dpi=300)
pltImpCat = plot_importance_by_category(imp)
pltImpCat.savefig("../figures/PWSDriversPaper/importanceCategories.jpeg", dpi=300)
ax = plot_importance_plants(imp)
pltPDP = plot_pdp(regrn, X_test)
pltImpCat.savefig("../figures/PWSDriversPaper/pdps.jpeg", dpi=300)
#to help interpret the pdps
#pltPairs = plot_scatter_feats(df_noSpec)

Run a few alternative versions of the RF model with reduced inputs
'''
print('only climate')
df_onlyClimate = df_noSpec.copy()
df_onlyClimate = df_onlyClimate[['pws','vpd_mean','ppt_cv','AI']]
X_test_oC, y_test_oC, regrn_oC, score_onlyClimate, imp_oC = regress(df_onlyClimate, optHyperparam=False)
print('only NDVI')
df_onlyPlant = df_noSpec.copy()
df_onlyPlant = df_onlyPlant[['pws','ndvi']]
X_test_oP, y_test_oP, regrn_oP, score_onlyPlant, imp_oP = regress(df_onlyPlant, optHyperparam=False)
print('only Soil')
df_onlySoil = df_noSpec.copy()
df_onlySoil = df_onlySoil[['pws','ks','bulk_density','Sr']]
X_test_oS, y_test_oS, regrn_oS, score_onlySoil, imp_oS = regress(df_onlySoil, optHyperparam=False)
print('only topo')
df_onlyTopo = df_noSpec.copy()
df_onlyTopo = df_onlyTopo[['pws','aspect','slope','twi']]
X_test_oT, y_test_oT, regrn_oT, score_onlyTopo, imp_oT = regress(df_onlyTopo, optHyperparam=False)
print('no VPD')
df_noVPD = df_noSpec.copy()
df_noVPD.drop('vpd_mean', axis = 1, inplace = True)
X_test_nV, y_test_nV, regrn_nV, score_noVPD, imp_nV = regress(df_noVPD, optHyperparam=False)
print('no NDVI')
df_noNDVI = df_noSpec.copy()
df_noNDVI.drop('ndvi', axis = 1, inplace = True)
X_test_nN, y_test_nN, regrn_nN, score_noNDVI, imp_nN = regress(df_noNDVI, optHyperparam=False)


'''
now check how explanatory power compares if don't have species vs. if have 
top 10 species one-hot-encoded
so want one run wSpec where have one-hot-encoded, don't drop species
and one run noSpec whre don't have species, but have same filterlist
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
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(pwsPred, pwsVec, s = 1, alpha = 0.4, color='k')
ax1.set_box_aspect(1)
ax1.set_xlabel("Predicted PWS", fontsize = 14)
ax1.set_ylabel("Actual PWS", fontsize = 14)
ax1.xaxis.set_ticks(np.arange(0, 6.2, 1))
ax1.yaxis.set_ticks(np.arange(0, 6.2, 1))
ax1.set_xlim(0,6), ax1.set_ylim(0,6)
ax1.set_title('Species mean', fontsize = 14)
ax1.annotate(f"R$^2$={coeffDeterm:0.2f}", (0.61,0.06),xycoords = "axes fraction", 
             fontsize=14, ha = "left")
ax1.annotate('a)', (-0.2,1.1),xycoords = "axes fraction", 
             fontsize=14, weight='bold')
ax2.set_box_aspect(1)
ax2.scatter(y_hat, y_test, s = 1, alpha = 0.4, color='k')
ax2.set_xlabel("Predicted PWS", fontsize = 14)
ax2.set_xlim(0,6), ax2.set_ylim(0,6)
ax2.xaxis.set_ticks(np.arange(0, 6.2, 1))
ax2.yaxis.set_ticks(np.arange(0, 6.2, 1))
ax2.set_title('Random forest', fontsize = 14)
ax2.annotate(f"R$^2$={score:0.2f}", (0.61,0.06),xycoords = "axes fraction", 
             fontsize=14, ha = "left")
ax2.annotate('b)', (-0.2,1.10),xycoords = "axes fraction", 
             fontsize=14, weight='bold')
fig.tight_layout()
plt.savefig("../figures/PWSDriversPaper/scatterPlotsModels.jpeg", dpi=300)
plt.show()

'''
dill.dump_session('./RFregression_dill.pkl')
'''