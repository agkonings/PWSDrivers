# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 07:44:09 2021

@author: konings
neural network regression of pws


    
"""
import os

import numpy as np
import pandas as pd
#from osgeo import gdal
import sklearn.ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import sklearn.inspection
from sklearn.inspection import permutation_importance
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches

import dirs

sns.set(font_scale = 1.5, style = "ticks")
plt.rcParams.update({'font.size': 18})

def cleanup_data(path):
    """
    path is where h5 file is stored
    """
    
    store = pd.HDFStore(path)
    df =  store['df']   # save it
    store.close()
    #df.drop(["lc","isohydricity",'root_depth', 'hft', 'p50', 'c', 'g1',"dry_season_length","lat","lon"],axis = 1, inplace = True)
    df.drop(["lc","vpd_mean","vpd_cv","ppt_mean","ppt_cv","ndvi","hft","sand",'vpd_cv',"ppt_lte_100","thetas","dry_season_length","t_mean","t_std","lat","lon"],axis = 1, inplace = True)
    #df.drop(["lc","ndvi","dry_season_length","lat","lon"],axis = 1, inplace = True)
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
    
    #scale inputs
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    '''
    mlp = MLPRegressor(max_iter = 500)
    #solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=32
    params = { 'hidden_layer_sizes': [(50,50,50), (50,100,50), (40,40,40,40), (100,100,100), (20,20,20)],
              'activation': ['logistic','tanh','relu'],
              'solver': ['sgd','adam'],
              'learning_rate': ['constant','adaptive'],
              'alpha': [0.00005, 0.001, 0.05]}
              ##'learning_rate': ['constant', 'adaptive'],
              ##'solver': ['lbfgs','sgd', 'adam'],
              ##'alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.05]}
    clf = GridSearchCV(mlp, param_grid=params) #TODO ADD CV PARAM
    clf.fit(X_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print('Best parameters found:\n', clf.best_params_)
    '''
    mlp = MLPRegressor(max_iter = 500, hidden_layer_sizes=(100,100,100), activation='tanh', 
                       solver='adam', learning_rate='adaptive', alpha=0.001)
    mlp.fit(X_train, y_train)
    score = mlp.score(X_test,y_test)
    
    
    print(f"[INFO] score={score:0.3f}")

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
    '''
    
    # assemble all importance with feature names and colors
    rImp = permutation_importance(mlp, X_test, y_test,
                            n_repeats=2, random_state=0)
    heights = rImp.importances_mean
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

    return X_test, y_test, mlp, score, imp


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
    
    
    
def main():
    plt.rcParams.update({'font.size': 18})

    #%% Load data
    path = os.path.join(dirs.dir_data, 'store_plant_soil_topo_climate_PWSthrough2021v3.h5')
    df = cleanup_data(path)
    
    #%% Train rf
    X_test, y_test, regrn, score,  imp = regress(df)  
     
    #%% make plots
    ax = plot_corr_feats(df)
    #still a bug somewhere in plot_error_pattern, ignroe for now
    #ax = plot_error_pattern(path, df)
    ax = plot_importance(imp)
    ax = plot_importance_by_category(imp)
    ax = plot_importance_plants(imp)
    ax = plot_preds_actual(X_test, y_test, regrn, score)
    plot_pdp(regrn, X_test)
    

if __name__ == "__main__":
    main()

