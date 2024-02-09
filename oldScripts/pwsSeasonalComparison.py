# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:58:46 2023

@author: konings
"""
import numpy as np
import pandas as pd
from osgeo import gdal
import os
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.plot import plotting_extent, show
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
import sklearn.preprocessing
import matplotlib.patches
import sklearn.inspection
from sklearn.inspection import permutation_importance
import sklearn.metrics
from sklearn.tree import plot_tree

import dirs

def plot_map(arrayToPlot, pwsExtent, stateBorders, title, vmin = None, vmax = None, clrmap = 'YlGnBu', savePath = None):
    '''make map with state borders'''
    
    #preliminary calculatios
    statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']    
    
    #actual plotting
    fig, ax = plt.subplots()
    if vmin != None and vmax != None:
        ax = rasterio.plot.show(arrayToPlot, vmin=vmin, vmax=vmax, extent=pwsExtent, ax=ax, cmap=clrmap)
    else:
        ax = rasterio.plot.show(arrayToPlot, extent=pwsExtent, ax=ax, cmap=clrmap)
    stateBorders[stateBorders['NAME'].isin(statesList)].boundary.plot(ax=ax, edgecolor='black', linewidth=0.5) 
    im = ax.get_images()[0]
    cbar = plt.colorbar(im, ax=ax) #ticks=range(0,6)
    #cbar.ax.set_xticklabels([ 'Deciduous','Evergreen','Mixed','Shrub','Grass', 'Pasture'])
    plt.title(title)
    ax.axis('off')
    plt.xticks([])
    plt.yticks([])
    if savePath != None:
        plt.savefig(savePath)
    plt.show() 
    
#load 
decThruMayPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_DecThruMay.tif'
pwsDecThruMayMap =rxr.open_rasterio(decThruMayPath, masked=True).squeeze()
junThruNovPath = 'G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_JunThruNov.tif'
#junThruNovPath = 'C:/repos/data/inputFeatures_wGNATSGO.h5'
pwsJunThruNovMap =rxr.open_rasterio(junThruNovPath, masked=True).squeeze()


#prep plotting
statesList = ['Washington','Oregon','California','Texas','Nevada','Idaho','Montana','Wyoming',
              'Arizona','New Mexico','Colorado','Utah']
ds = gdal.Open(decThruMayPath)
geotransform = ds.GetGeoTransform()
statesPath = "C:/repos/data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp"
states = gpd.read_file(statesPath)    
#Map = df_to_raster(shapdf['shap_vanGen_n'], np.shape(pwsDecThruMayRaster), lat, lon, geotransform)    

pwsExtent = plotting_extent(pwsDecThruMayMap, pwsDecThruMayMap.rio.transform())
plot_map(pwsDecThruMayMap, pwsExtent, states, 'PWS, Dec through May', vmin = 0, vmax = 2)
plot_map(pwsJunThruNovMap, pwsExtent, states, 'PWS, Jun through Nov', vmin = 0, vmax = 2)
diffMap = pwsDecThruMayMap - pwsJunThruNovMap
plot_map(diffMap, pwsExtent, states, 'PWS, Jun through Nov', vmin = -1, vmax = 1)

#calculate cross-correlations
def load_pws_array(pwsPath):
    dspws = gdal.Open(pwsPath)
    gtpws= dspws.GetGeoTransform()
    arraypws = np.array(dspws.GetRasterBand(1).ReadAsArray())
    
    return arraypws

decThruMayArray = load_pws_array(decThruMayPath)
junThruNovArray = load_pws_array(junThruNovPath)
dfPWS = pd.DataFrame()
dfPWS['decThruMay'] = decThruMayArray.flatten()
dfPWS['junThruNov'] = junThruNovArray.flatten()

#calculate cross-correlations but also remove points were don't have features
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

def cleanup_data(df, droppedVarsList):
    """
    path : where h5 file is stored
    droppedVarsList : column names that shouldn't be included in the calculation
    """
    
    df.drop(droppedVarsList, axis = 1, inplace = True)
    df.dropna(inplace = True)
    lat = df["lat"]
    lon = df["lon"]
    df.drop(["lat", "lon"], axis=1, inplace=True)
    df.reset_index(inplace = True, drop = True)
    
    return df, lat, lon
        
featPath = os.path.join('G:/My Drive/0000WorkComputer/dataStanford/PWSFeatures/inputFeatures.h5')
dfDecThruMay =  load_data(featPath, decThruMayPath)
dfJunThruNov =  load_data(featPath, junThruNovPath)

droppedVarsList = ["twi","thetas","ks","silt","clay","slope","species","vpd_mean","vpd_cv","ppt_mean","ppt_cv","ndvi",'vpd_cv',"ppt_lte_100","dry_season_length","t_mean","t_std","Sr","Sbedrock"]
dfDecThruMay, lat_DtM, lon_DtM = cleanup_data(dfDecThruMay, droppedVarsList)
dfJunThruNov, lat_JtN, lon_JtN = cleanup_data(dfJunThruNov, droppedVarsList)
#pull out same values
commonInd = list(set(dfDecThruMay.index) & set(dfJunThruNov.index))
pearsonr(dfDecThruMay.pws[commonInd].values, dfJunThruNov.pws[commonInd].values)

'''
At this point, super confused about why there's no seasonal difference
debug the actual regression and imp generation
Since having weird debugging issues don't put behind a function
'''
leaves = 4
decrease = 1e-8


X_DtM = dfDecThruMay.drop("pws",axis = 1)
y_DtM = dfDecThruMay['pws']
# separate into train and test set
X_DtM_train, X_DtM_test, y_DtM_train, y_DtM_test = sklearn.model_selection.train_test_split(
    X_DtM, y_DtM, test_size=0.33, random_state=32)
#can get highest with 3 leaves, 120 nEst, decrease 1e-8, but that seems like low number of leaves
#old configuration was leaves = 6, decrease 1e-6, nEst = 50
# construct rf model
regrn_DtM = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                  min_impurity_decrease=decrease, n_estimators = 90)
# train
regrn_DtM.fit(X_DtM_train, y_DtM_train)
# test set performance
score_DtM = regrn_DtM.score(X_DtM_test,y_DtM_test)
# assemble all importance with feature names and colors
rImp_DtM = permutation_importance(regrn_DtM, X_DtM_test, y_DtM_test,
                        n_repeats=5, random_state=8)
heights_DtM = rImp_DtM.importances_mean
uncBars_DtM = rImp_DtM.importances_std
ticks_DtM = X_DtM.columns
imp_DtM = pd.DataFrame(index = ticks_DtM, columns = ["importance"], data = heights_DtM)
imp_DtM['importance std'] = uncBars_DtM
print('tot hier')

#now repeat for June through November
X_JtN = dfJunThruNov.drop("pws",axis = 1)
y_JtN = dfJunThruNov['pws']
# separate into train and test set
X_JtN_train, X_JtN_test, y_JtN_train, y_JtN_test = sklearn.model_selection.train_test_split(
    X_JtN, y_JtN, test_size=0.33, random_state=32)
#can get highest with 3 leaves, 120 nEst, decrease 1e-8, but that seems like low number of leaves
#old configuration was leaves = 6, decrease 1e-6, nEst = 50
# construct rf model
regrn_JtN = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                  min_impurity_decrease=decrease, n_estimators = 90)
# train
regrn_JtN.fit(X_JtN_train, y_JtN_train)
# test set performance
score_JtN = regrn_JtN.score(X_JtN_test,y_JtN_test)
# assemble all importance with feature names and colors
rImp_JtN = permutation_importance(regrn_JtN, X_JtN_test, y_JtN_test,
                        n_repeats=5, random_state=8)
heights_JtN = rImp_JtN.importances_mean
uncBars_JtN = rImp_JtN.importances_std
ticks_JtN = X_JtN.columns
imp_JtN = pd.DataFrame(index = ticks_JtN, columns = ["importance"], data = heights_JtN)
imp_JtN['importance std'] = uncBars_JtN

#then compare, numbers on imp_JtN and imp_JtN
def df_to_raster(dfColumn, rasterShape, lat, lon, geotransform):
    '''
    Take a dataframe and corresponding lat and lon values of same size and turn into raster
    '''

    valMap = np.empty( rasterShape ) * np.nan    
    latInd = np.round( (lat.to_numpy() - geotransform[3])/geotransform[5] ).astype(int)
    lonInd = np.round( (lon.to_numpy() - geotransform[0])/geotransform[1] ).astype(int)
    valMap[latInd, lonInd] = dfColumn

    return valMap 

pred_DtM = regrn_DtM.predict(X_DtM)
pred_JtN = regrn_JtN.predict(X_JtN)

#stuff needed for plotting
ds = gdal.Open(decThruMayPath)
geotransform = ds.GetGeoTransform()
pwsRaster = rxr.open_rasterio(decThruMayPath, masked=True).squeeze()

pred_DtMMap = df_to_raster(pred_DtM, np.shape(pwsRaster), lat_DtM, lon_DtM, geotransform)    
pred_JtNMap = df_to_raster(pred_JtN, np.shape(pwsRaster), lat_JtN, lon_JtN, geotransform)    

#plot predictions for each case
plot_map(pred_DtMMap, pwsExtent, states, 'Predictions, Dec through May', vmin = 0, vmax = 2)
plot_map(pred_JtNMap, pwsExtent, states, 'Predictions, Jun through Nov', vmin = 0, vmax = 2)
#plot difference in predictions
diffMap = pred_DtMMap - pred_JtNMap
plot_map(diffMap, pwsExtent, states, 'Predictions difference', vmin = -1, vmax = 1)

#score = regrn.score(X_test,y_test)

'''
#try one more, what happens if re-arrange columns
df2 = dfJunThruNov.copy() 
df2 = df2[['pws','aspect','agb','dist_to_water','nlcd','elevation',
          'g1','isohydricity','root_depth','canopy_height','p50','gpmax','c',
          'vanGen_n','AI']]
#re-run regression!
X2 = df2.drop("pws",axis = 1)
y2 = df2['pws']
# separate into train and test set
X2_train, X2_test, y2_train, y2_test = sklearn.model_selection.train_test_split(
    X2, y2, test_size=0.33, random_state=32)
#can get highest with 3 leaves, 120 nEst, decrease 1e-8, but that seems like low number of leaves
#old configuration was leaves = 6, decrease 1e-6, nEst = 50
# construct rf model
regrn2 = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=leaves, \
                  min_impurity_decrease=decrease, n_estimators = 90)
# train
regrn2.fit(X2_train, y2_train)
# test set performance
score2 = regrn2.score(X2_test,y2_test)
# assemble all importance with feature names and colors
rImp2 = permutation_importance(regrn2, X2_test, y2_test,
                        n_repeats=3, random_state=8)
heights2 = rImp2.importances_mean
uncBar2 = rImp2.importances_std
ticks2 = X2.columns
imp2 = pd.DataFrame(index = ticks2, columns = ["importance"], data = heights2)
imp2['importance std'] = uncBar2
'''

#plot some heatmaps to see if that explains the patterns
fig, ax = plt.subplots()
ax.hist2d(dfJunThruNov['vanGen_n'],dfJunThruNov['pws'], (75,75) )
plt.xlabel('Van Genuchten n'), plt.ylabel('PWS')
plt.show()
fig, ax = plt.subplots()
ax.hist2d(dfJunThruNov['elevation'],dfJunThruNov['pws'], (75,75) )
plt.xlabel('Elevation'), plt.ylabel('PWS')
plt.show()

#plot histograms of different variables
def calcNormalizedHist(data):
    zData = ( data - np.mean(data) )/ np.std(data)
    h, edges = np.histogram(zData, bins=80, range=(-3.1,4))
    return h, edges

hPWS, edgesPWS = calcNormalizedHist(dfJunThruNov['pws'])
hGen, edgesGen = calcNormalizedHist(dfJunThruNov['vanGen_n'])
hElev, edgesElev = calcNormalizedHist(dfJunThruNov['elevation'])
hAI, edgesAI = calcNormalizedHist(dfJunThruNov['AI'])
fig, ax = plt.subplots()
edgeCent = (edgesPWS[0:-1] + edgesPWS[1:] ) / 2
ax.step(edgeCent, hPWS, label='PWS')
ax.step(edgeCent, hGen, label='van Genuchten n')
ax.step(edgeCent, hElev, label='Elevation')
ax.step(edgeCent, hAI, label='Aridity Index')
plt.xlabel('z-score')
plt.legend(loc='upper right')
plt.show()

#re-confirm anew: cross-correlations between elevation, AI, vanGen_n and PWS are minimal, but
#vanGen_n builds great model and others don't
def testOneVar(dataPWS, dataX, name):
    #re-run regression!
    #X = df2.drop("pws",axis = 1)
    #y2 = df2['pws']
    # separate into train and test set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        dataX, dataPWS, test_size=0.33, random_state=32)
    #can get highest with 3 leaves, 120 nEst, decrease 1e-8, but that seems like low number of leaves
    #old configuration was leaves = 6, decrease 1e-6, nEst = 50
    # construct rf model
    regrn = sklearn.ensemble.RandomForestRegressor(min_samples_leaf=2, \
                                            min_impurity_decrease=decrease, n_estimators = 90)
    # train
    regrn.fit(X_train, y_train)
    # test set performance
    scoreTest = regrn.score(X_test,y_test)
    scoreTrain = regrn.score(X_train,y_train)
    # assemble all importance with feature names and colors
    rImp = permutation_importance(regrn, X_test, y_test,
                        n_repeats=3, random_state=8)
    
    #fig = plt.figure()
    #plot_tree(regrn.estimators_[0], 
    #          feature_names=dataX.columns,
    #          filled=True, impurity=True, 
    #          rounded=True)
        
    print('RF test score between PWS & ' + name + ': ' + str(scoreTest) )
    print('RF train score between PWS & ' + name + ': ' + str(scoreTrain) )
    print('RF importance between PWS & ' + name + ': ' + str(rImp.importances_mean[0]) )
    print('RF importance between PWS & TWI: ' + str(rImp.importances_mean[1]) )
    print('Pearson r between PWS & ' + name + ': ' + str(pearsonr(dataPWS,dataX[dataX.columns[0]])[0]) )
    print('Spearman r between PWS & ' + name + ': ' + str(spearmanr(dataPWS,dataX[dataX.columns[0]])[0]) )
    
testOneVar(dfJunThruNov['pws'], dfJunThruNov[['vanGen_n','dist_to_water']], 'van Genuchten n')
dfJunThruNov['vanGen_n2'] = dfJunThruNov['vanGen_n'].to_numpy() + 0.0001*np.random.rand(192765)
print('After adding random numbers')
testOneVar(dfJunThruNov['pws'], dfJunThruNov[['vanGen_n2','dist_to_water']], 'van Genuchten n')
testOneVar(dfJunThruNov['pws'], dfJunThruNov[['elevation', 'dist_to_water']], 'elevation')
testOneVar(dfJunThruNov['pws'], dfJunThruNov[['AI', 'dist_to_water']], 'AI')