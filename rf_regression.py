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
    
    dspws = gdal.Open(os.path.join(dirs.dir_data, "pws_features","PWS_through2021.tif"))
    gtpws= dspws.GetGeoTransform()
    arraypws = np.array(dspws.GetRasterBand(1).ReadAsArray())
    
    df['pws'] = arraypws.flatten()
    
    #re-arrange columns so pws goes first
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
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
    soil = ['clay', 'silt','thetas', 'ks', 'vanGen_n','Sr','Sbedrock']
    climate = [ 'dry_season_length', 'vpd_mean', 'vpd_cv',"ppt_mean","ppt_cv","t_mean","t_std","ppt_lte_100","AI"]
    topo = ['elevation', 'aspect', 'slope', 'twi',"dist_to_water"]
    traits = ['isohydricity', 'root_depth', 'p50', 'gpmax', 'c', 'g1']
    
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
                 "gpmax":"K_{max,x}",
                 "c":"Capacitance",
                 "g1":"g$_1$",
                 "n":"$n$",
                 "nlcd": "land cover",
                 "aspect":"Aspect",
                 "slope":"Slope",
                 "twi":"TWI",
                 "ppt_lte_100":"Dry months",
                 "dist_to_water":"Dist to water",
                 "t_mean":"Temp$_{mean}$",
                 "t_std":"Temp$_{st dev}$",
                 "lon":"Lon", "lat":"Lat",
                 "vanGen_n":"van Genuchten n",
                 "AI":"Aridity Index",
                 "species":"species",
                 "species_64.0":"Species 64",
                 "species_65.0":"Species 65",
                 "species_69.0":"Species 69",
                 "species_106.0":"Species 106",
                 "species_108.0":"Species 108",
                 "species_122.0":"Species 122",
                 "species_133.0":"Species 133",
                 "species_202.0":"Species 202",
                 "species_746.0":"Species 746",
                 "species_756.0":"Species 756",
                 "species_814.0":"Species 814"
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
    
    
    
plt.rcParams.update({'font.size': 18})


#%% Load data
path = os.path.join(dirs.dir_data, 'inputFeatures.h5')
store = pd.HDFStore(path)
df =  store['df']   # save it
store.close()

#add particular PWS file
pwsPath = os.path.join(dirs.dir_data, "pws_features","PWS_through2021.tif") 
df = add_pws(df, pwsPath)

#oldList: ["lc","isohydricity",'root_depth', 'hft', 'p50', 'c', 'g1',"dry_season_length","lat","lon"],axis = 1, inplace = True)
droppedVarsList = ["species","AI","vpd_mean","vpd_cv","ppt_mean","ppt_cv","ndvi",'vpd_cv',"ppt_lte_100","thetas","dry_season_length","t_mean","t_std","lat","lon","Sr","Sbedrock"]
#oldList: (["lc","ndvi","dry_season_length","lat","lon"],axis = 1, inplace = True)
df = cleanup_data(df, droppedVarsList)
error

#%% Train rf
X_test, y_test, regrn, score,  imp = regress(df)  
 
#%% make plots
ax = plot_corr_feats(df)
#still a bug somewhere in plot_error_pattern, ignroe for now
#ax = plot_error_pattern(path, df)
ax = plot_importance(imp)
ax = plot_importance_by_category(imp)
ax = plot_importance_plants(imp)
x = plot_preds_actual(X_test, y_test, regrn, score)

print('now doing species power calculations')    
'''now check how explanatory power compares if don't have species vs. if have 
top 10 species one-hot-encoded
so want one run wSpec where have one-hot-encoded, don't drop species
and one run noSpec whre don't have species, but have same filterlist'''

#common species list hand-calculated separately based on most entries in species column
#shorter list gets about 0.03 better r2 because have more data, so use that as in between point
#between enough data to do well and not too many columns
#commonSpecList = {65, 69, 122, 202, 756, 64, 106, 108, 133, 746, 814}
commonSpecList = {65, 69, 122, 202, 756, 64, 106, 108}

#first calculate with species (one-hot encoded)
print('with species one-hot encoded') 
droppedVarsList.remove('species') #so don't drop from list
df_wSpec = cleanup_data(path, droppedVarsList)
noDataRows = df_wSpec.loc[~df_wSpec.species.isin(commonSpecList)]
df_wSpec.drop(noDataRows.index, inplace=True)
df_w1SpecCol = df_wSpec.copy()
df_wSpec = pd.get_dummies(df_wSpec, columns=['species'])
X_test_wSpec, y_test_wSpec, regrn_wSpec, score_wSpec,  imp_wSpec = regress(df_wSpec) 
ax = plot_importance(imp_wSpec)

#calculate RF performance on same points, but without species info
print('without species info') 
df_noSpec = df_w1SpecCol.drop('species', axis=1, inplace=False)
X_test_noSpec, y_test_noSpec, regrn_noSpec, score_noSpec,  imp_noSpec = regress(df_noSpec) 
ax = plot_importance(imp_noSpec)

#instead, for each unique species, calculate a predicted value
'''print('predictive ability with all species') 
pwsVec = df_w1SpecCol['pws']
specVec = df_w1SpecCol['species']
pwsPred = np.zeros(pwsVec.shape)
specNumbers = {}
for specCode in np.unique(df_wSpec['species']):
    specNumbers[specCode]=np.sum(specVec == specCode)
    #differentiating (obs_i-X)^2 shows that optimal predictor is mean of each cat
    thisMean = np.mean( pwsVec[specVec == specCode] )
    pwsPred[specVec == specCode] = thisMean

resPred = pwsVec - pwsPred
SSres = np.sum(resPred**2)
SStot = np.sum( (pwsVec - np.mean(pwsVec))**2 ) #total sum of squares
coeffDeterm = 1 - SSres/SStot'''

print('predictive ability with top species') 
pwsVec = df_w1SpecCol['pws']
specVec = df_w1SpecCol['species']
pwsPred = np.zeros(pwsVec.shape)
specNumbers = {}
for specCode in np.unique(df_w1SpecCol['species']):
    specNumbers[specCode]=np.sum(specVec == specCode)
    #differentiating (obs_i-X)^2 shows that optimal predictor is mean of each cat
    thisMean = np.mean( pwsVec[specVec == specCode] )
    pwsPred[specVec == specCode] = thisMean

resPred = pwsVec - pwsPred
SSres = np.sum(resPred**2)
SStot = np.sum( (pwsVec - np.mean(pwsVec))**2 ) #total sum of squares
coeffDeterm = 1 - SSres/SStot

print('amount explained with ONLY species info ' + str(coeffDeterm))
print('fraction explained by species' + str(coeffDeterm/score_wSpec))

'''NBcoeffDeterm was about 0.15 with all species, now 0.12 with top species'

summary findings = scores very similar!! 0.438 vs 0.436 with or without species.
so doesn't help much. coeffDeterm = 0.14 so species alone can get about 1/3rd of var
That is, R2 about 0.12 with categorical prediction from species vs about 0.45 with RF
note also that while there are 199 species, there's a ton that barely have any representation
most common are: spec 65 (n=2256), spec 69 (n=1284), spec 122 (2589), spec202 (1631)
spec 756 (3151). So the top 5 species adds up to 10911 out of 19551 pixels (56%)
if add a bit more, we also have spec 58 (405), spec 64 (660), spec 106 (561), spec 108 (753),
spec 133 (492), spec 746 (486), spec 814 (465)
top 8 species gets 12885 or 66%
so toop 11 species gets 14241, or 73%
so could do 7-way one-hot encoding as additional test and compairson of interations'''