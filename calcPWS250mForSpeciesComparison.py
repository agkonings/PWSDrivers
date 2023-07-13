# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:27:24 2023

@author: konings
"""

import os
import datetime
import sys

from osgeo import gdal, osr
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
import pickle

def get_dates(date, maxLag = 6):
    subsetDates = []
    for delta in range(1, maxLag):
        shiftedDate = pd.to_datetime(date,format = "%Y-%m-%d") - DateOffset(months = delta)
        shiftedDate = shiftedDate.date().strftime(format = "%Y-%m-%d")                    
        for day in [1,15]:
            subsetDates+= [shiftedDate[:-2]+"%02d"%day]
    
    subsetDates = pd.to_datetime(subsetDates,format = "%Y-%m-%d")
    subsetDates = list(subsetDates.sort_values(ascending = False).strftime(date_format = "%Y-%m-%d") )
    
    return subsetDates   

def regress(df,norm = "lfmc_dfmc_norm", coefs_type = "unrestricted"):            
    cols = [col for col in df.columns if "dfmc" in col]        
    X = df.loc[:,cols]
    y = df.iloc[:,0] ### 
    if norm=="lfmc_norm":
        y = (y-y.mean())/y.std()
    elif norm=="dfmc_norm":
        X = (X - X.mean())/X.std()
    elif norm == "lfmc_dfmc_norm":
        y = (y-y.mean())/y.std()
        X = (X - X.mean())/X.std()
    
    if coefs_type=="positive":
        reg = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random').fit(X,y)
    else:
        reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    coefs = [reg.intercept_]+list(reg.coef_)
    
    return r2, coefs, df['lon'].iloc[0],df['lat'].iloc[0]  

#load master

#find row/rat with master DFMC that is most specific
#mhave dominantLocs save HDF

#but want LFMC!

#to be efficient, create master similar to original PWS but do at point scale
#will have mismatch between resolution of DFMC and LFMC

#so load lats and lons from rf_regression/df_wSpec pickle
FIA_wSpec = pd.read_pickle('../data/df_wSpec.pkl')

#load master file from the 4 km PWS calculation to be able to pull the DFMC from there
#### FILL IN HERE FILL IN HERE FILL IN HERE FILL IN HERE FILL IN HERE FILL IN HERE FILL IN HERE 

#current format is such that each master row is one point in time and one location
#keep it that way so can re-use 
#Krishan does it one day at a time
#here, repat, but instead of adding all locations add a loop where you pull

dir_4km = "G:/My Drive/0000WorkComputer/dataStanford/lfmc_dfmc_anomalies/"
dir_250m ='F:/lfmc_from_sar_data/lfmcKrishna/'

#make list of dates used in regression
years = range(2016, 2022) #[inclusive, exclusive)
months = range(1,13) #months = list(range(12,13)) + list(range(1,6))
days = [1,15]

dates = []
for year in years:
    for month in months:
        if year == 2016 and month <6: # first 6 months ignored because DFMC is lagged by 5 months
            continue
        for day in days:
            dates+=["%s-%02d-%02d"%(year, month, day)]


'''
Actually now create master file to do the regression on, following 
the logic and code in wildfire_from_lfmc/analysis/pws_calculation
'''
#load geotransforms at 4km and at 250m
filename_4km = os.path.join(dir_4km, "lfmc_map_2017-01-15.tif")
ds_4km = gdal.Open(filename_4km)    
gt_4km = ds_4km.GetGeoTransform()
ds_4km = None    
filename_250m = os.path.join(dir_250m, "lfmc_map_2017-01-15.tif")
ds_250m = gdal.Open(filename_250m)    
gt_250m = ds_250m.GetGeoTransform()
ds_250m = None
        
#Find x_index_4km, y_index_4km values for FIA plots being studied
x_index_4km = np.floor( (FIA_wSpec['lon']-gt_4km[0])/gt_4km[1] ).to_numpy().astype(int)
y_index_4km = np.floor( (FIA_wSpec['lat']-gt_4km[3])/gt_4km[5] ).to_numpy().astype(int)

#Find x_index_250m, y_index_250m values for FIA plots being studied
x_index_250m = np.floor( (FIA_wSpec['lon']-gt_250m[0])/gt_250m[1] ).to_numpy().astype(int)
y_index_250m = np.floor( (FIA_wSpec['lat']-gt_250m[3])/gt_250m[5] ).to_numpy().astype(int)

#create master with all DFMC loaded but without LFMC
master = pd.DataFrame()
for date in dates:
    df = pd.DataFrame()
    filename_4km = os.path.join(dir_4km, "lfmc_map_%s.tif"%date)
    ds_4km = gdal.Open(filename_4km)    
    #dfmc: raster band 2 is for 100 hr rather than 1000 hr fuels
    dfmc = np.array(ds_4km.GetRasterBand(2).ReadAsArray())
    ds_4km = None    
    filename_250m = os.path.join(dir_250m, "lfmc_map_%s.tif"%date)
    ds_250m = gdal.Open(filename_250m)    
    if ds_250m == None:
        print('Missing at ' + date)
        continue
    lfmc = np.array(ds_250m.GetRasterBand(1).ReadAsArray()).astype(float)
    lfmc[lfmc==-9999] = np.nan
    
    ds_250m = None
        
    df['lfmc(t)'] = lfmc[y_index_250m, x_index_250m]
    df['dfmc(t)'] = dfmc[y_index_4km, x_index_4km]
    df['pixel_index'] = df.index
    df['lat'] = FIA_wSpec['lat']
    df['lon'] = FIA_wSpec['lon']    
        
    df['date'] = date
    ctr = 1
    sys.stdout.write('\r'+'[INFO] Time step %s'%date)
    sys.stdout.flush()
    # print(date)
    subsetDates = get_dates(date, maxLag = 6)
    for t in subsetDates:
        shiftedFile = os.path.join(dir_4km, "lfmc_map_%s.tif"%t)
        ds = gdal.Open(shiftedFile)
        if ds == None:
            continue
        else:
            df['dfmc(t-%d)'%ctr] = np.array(ds.GetRasterBand(2).ReadAsArray())[y_index_4km, x_index_4km]
            ctr+=1
    df.dropna(inplace = True)
    master = master.append(df,ignore_index = True)     
master = master.dropna()


'''Now do actual regression'''
print('\n Regressing...')
# Remove pixels which have less than 25 data points
master = master.groupby("pixel_index").filter(lambda df: df.shape[0] > 25)

#%% Calculate PWS
# print('\r')
print('[INFO] Regressing')
out = master.groupby('pixel_index').apply(regress,norm = "lfmc_dfmc_norm", coefs_type = "positive")

coefs = [x[1] for x in out]
x_loc = [x[2] for x in out]
y_loc = [x[3] for x in out]
PWS_250m = [np.sum(x[1:]) for x in coefs] #sum of ocefficients

#turn into dataframe
data = {'PWS': PWS_250m, 'lat': y_loc, 'lon': x_loc}
df = pd.DataFrame(data)

pickleLoc = '../data/PWS_250m.pkl'
with open(pickleLoc, 'wb') as file:
    pickle.dump(df, file)