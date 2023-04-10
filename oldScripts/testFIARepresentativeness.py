# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:08:51 2022

@author: konings
This script aims to calculate the number of plots under different thresholds of
specis dominance at a given site or not

 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

def calcDominantLocs(combinedDf, dominantThresh):
    '''
    Find only the locations with a dominant fraction of total basal area from
    one species (above some threshold fraction dominantThresh) and create
    a dataframe that stores the fraction of the basal area from that location
    associated iwth the dominatn species, as well as the species 
    '''
    dominantLocs = pd.DataFrame()
    noGoodCnt = 0
    for unSite in combinedDf['siteID'].unique(): 
        sitedf = combinedDf[combinedDf['siteID'] == unSite]
        #if only species listed, assume dominates
        if len(sitedf) == 1:
            dominantLocs = pd.concat([dominantLocs, sitedf])
        else:
            #if multiple years of meas at same site, pick the most recent
            if sitedf['INVYR'].nunique() > 1:
                sitedf = sitedf[sitedf['INVYR'] == sitedf['INVYR'].max()]
            #if mulitple condition IDs, pick the lowest
            if sitedf['CONDID'].nunique() > 1:
                sitedf = sitedf[sitedf['CONDID'] == sitedf['CONDID'].min()]
            #note that if there's two sites within 4 km, they can have the same 
            #species across the two sites, and harder to check which is dominant
            #check for this and treat the whole set-up differently then
            if sitedf['SPCD'].nunique() < len(sitedf):
                foundSite = False
                #could probably replace this loop with groupby 
                #priortize programmer time for nwo
                for unSpecies in sitedf['SPCD'].unique():
                    theseSpec = sitedf[sitedf['SPCD'] == unSpecies]            
                    BARat = theseSpec['BALiveAc'].sum()/sitedf['BALiveAc'].sum()
                    if BARat > dominantThresh:
                        #have two locations now, but otherwise identical for our
                        #later purpose. So just add one randomly 
                        dominantLocs = pd.concat([dominantLocs, theseSpec.head(1)])
                        foundSite = True
                #if you make it through this loop, no species is dominant
                if foundSite == False:
                    noGoodCnt += 1
            #so have multiple species possible, but only one site. 
            else:
                BARat = sitedf['BALiveAc']/sitedf['BALiveAc'].sum()
                #check if one species is dominant in terms of basal area
                if dominantThresh>0 and BARat.max()>dominantThresh: 
                    if len(sitedf[BARat>dominantThresh])>1:
                        raise Exception('Multiple dominant species are impossible')
                    dominantLocs = pd.concat([dominantLocs, sitedf[BARat>dominantThresh]])
                else:
                    noGoodCnt += 1
                    
    return dominantLocs

#notes: seem to have about 138,000 siteID unique values according to quick'n dirty calcs
#that's roguhly the same number of entries in FIA_traits_Alex.
#also seems like you can find one plot ID # that corresponds to tally different (10 degrees apart!)
#lat and lon dependign on year. So that value is basically useless.
# for now just make site index with year and sort on unique values for those

def calcUniqueLocs(combinedDf):
    '''
    Find only the unique locations, so remove older years if multiple years
    then get summed BA at each site
    '''
    locs = pd.DataFrame()
    noGoodCnt = 0
    for unSite in combinedDf['siteID'].unique(): 
        sitedf = combinedDf[combinedDf['siteID'] == unSite]
        #if only species listed, assume dominates
        if len(sitedf) == 1:
            BATot = sitedf['BALiveAc']
            sitedf = sitedf.iloc[0]
            sitedf['BATot'] = BATot
        else:
            #if multiple years of meas at same site, pick the most recent
            if sitedf['INVYR'].nunique() > 1:
                sitedf = sitedf[sitedf['INVYR'] == sitedf['INVYR'].max()]
            #if mulitple condition IDs, pick the lowest
            if sitedf['CONDID'].nunique() > 1:
                sitedf = sitedf[sitedf['CONDID'] == sitedf['CONDID'].min()]

            BATot = sitedf['BALiveAc'].sum()
            sitedf = sitedf.iloc[0, :]
            sitedf['BATot'] = BATot
            
        locs = pd.concat([locs, sitedf])
                    
    return locs, noGoodCnt

flpth = 'G:/My Drive/0000WorkComputer/dataStanford/fiaAnna/'

#read two csv files
varsdf = pd.read_csv(flpth + 'CONDVars_LL.csv', sep=',', header=0, 
                     usecols=['CN','INVYR', 'PLOT', 'MEASYEAR', 'CONDID', \
					 'LAT', 'LON', 'FORTYPCD', 'FLDTYPCD', 'CONDPROP_UNADJ','BALIVE'])       
spdf = pd.read_csv(flpth + 'COND_Spp_Live_Ac.csv', sep=',', header=0, 
                     usecols=['COND_CN','SPCD','BALiveAc','TPALiveAc'])
p50df = pd.read_csv(flpth + 'FIA_traits_Alex.csv', sep=',', header=0)

#varsdf has 1,084,063 unique CN values, out of the same number of rows
#spdf has 413,987 unique COND_CN values, out of 1,926,679

#merge dataset to add lat lon information to condition information
combdf = spdf.merge(varsdf, left_on=['COND_CN'], right_on=['CN'], how='inner')
print( 'Minimum lat: ' + str(np.min(combdf['LAT'])) )
print( 'Maximum lat: ' + str(np.max(combdf['LAT'])) )
print( 'Minimum lon: ' + str(np.min(combdf['LON'])) )
print( 'Maximum lon: ' + str(np.max(combdf['LON'])) )

#Load PWS lats and lons, geotransform to figure out how to aggregate
ds = gdal.Open('C:/repos/data/pws_features/PWS_through2021_allSeas.tif')
gt = ds.GetGeoTransform()
pws = ds.GetRasterBand(1).ReadAsArray()
pws_y,pws_x = pws.shape
wkt_projection = ds.GetProjection()
ds = None

#just brute force search for locations where one species is dominant
#can't do this by condition because identical plots measured 10 years apart will 
#have different condition numbers.
#instead, approach by making cheap site index that maps to each location within 4km
#1/gt is actually 27.83 instead of 28, but assume this is close enough 
lonInd = np.floor( (combdf['LON']-gt[0])/gt[1] ).to_numpy().astype(int)
latInd = np.floor( (combdf['LAT']-gt[0])/gt[1] ).to_numpy().astype(int)
#latInd = np.floor(combdf['LAT']*27)
cheapSiteID = lonInd*1e5 + latInd 
combdf['siteID'] = cheapSiteID


#here is the brute forcing. When you have a site-index
#if two latitudes, add basal areas per species...whole mess
#if only one, pick latest year
dominantThresh = 0.7
dominantLocs = calcDominantLocs(combdf, dominantThresh)
debugLocs = dominantLocs.copy()

# filter for points with locations in PWS grid
#for each dominantLoc, find grid index in PWS
indX = np.floor( (dominantLocs['LON']-gt[0])/gt[1] ).to_numpy().astype(int)
indY = np.floor( (dominantLocs['LAT']-gt[3])/gt[5] ).to_numpy().astype(int)
n1, n2 = pws.shape
#this has pixels in American Samoa and the like. Remove those
mask = np.ones(indX.shape, dtype=bool)
mask[indX < 0] = False
mask[indX >= n2] = False
mask[indY < 0] = False
mask[indY > n1] = False
indX = indX[mask]
indY = indY[mask]
dominantLocs = dominantLocs[mask]

#filter sites without PWS values (e.g. northeast of domain states)
dominantLocs2 = dominantLocs.loc[~np.isnan(pws[indY,indX]), :]
BA_domFIA = np.sum(dominantLocs2['BALiveAc'])


''' 
Now repeat to just get the most recent site
'''
dominantThresh = 0
#allLocs, nNoGood = calcUniqueLocs(combdf)
locs = pd.DataFrame()
for unSite in combdf['siteID'].unique(): 
    sitedf = combdf[combdf['siteID'] == unSite]
    #if only species listed, assume dominates
    if len(sitedf) == 1:
        BATot = sitedf['BALiveAc']
        sitedf['BATot'] = BATot
    else:
        #if multiple years of meas at same site, pick the most recent
        if sitedf['INVYR'].nunique() > 1:
            sitedf = sitedf[sitedf['INVYR'] == sitedf['INVYR'].max()]
        #if mulitple condition IDs, pick the lowest
        if sitedf['CONDID'].nunique() > 1:
            sitedf = sitedf[sitedf['CONDID'] == sitedf['CONDID'].min()]
        
        BATot = sitedf['BALiveAc'].mean()
        #arbitrarily select minium species code
        sitedf = sitedf[sitedf['SPCD'] == sitedf['SPCD'].min()]
        sitedf['BATot'] = BATot
        
        
    locs = pd.concat([locs, sitedf])
    
allLocs = locs
print(str(allLocs.shape))

# filter for points with locations in PWS grid
#for each dominantLoc, find grid index in PWS
allIndX = np.floor( (allLocs['LON']-gt[0])/gt[1] ).to_numpy().astype(int)
allIndY = np.floor( (allLocs['LAT']-gt[3])/gt[5] ).to_numpy().astype(int)
#this has pixels in American Samoa and the like. Remove those
allMask = np.ones(allIndX.shape, dtype=bool)
allMask[allIndX < 0] = False
allMask[allIndX >= n2] = False
allMask[allIndY < 0] = False
allMask[allIndY > n1] = False
allIndX = allIndX[allMask]
allIndY = allIndY[allMask]
allLocs = allLocs[allMask]

#filter sites without PWS values (e.g. northeast of domain states)
allLocs2 = allLocs.loc[~np.isnan(pws[allIndY,allIndX]), :]
BA_allFIA = np.sum(allLocs2['BALiveAc'])

print('Fraction of basal area captured : ' + str(BA_domFIA/BA_allFIA))


'''
Note that there still seems something wrong with the code aobve

#can we just start from scratch?
#1) we have BA_locations
#2) for everything else, for each siteit, find the mot recent year
#3) create dataframe with only location, lattiude, siteid, year, mean BA
#4) filter to PWS grid
#4) is this worth ca

But do we even need to do this? 
