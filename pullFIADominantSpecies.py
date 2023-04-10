# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:08:51 2022

@author: konings
TODO CHECK 0.7!!!!!!!!!!!!!!!
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

#notes: seem to have about 138,000 siteID unique values according to quick'n dirty calcs
#that's roguhly the same number of entries in FIA_traits_Alex.
#also seems like you can find one plot ID # that corresponds to tally different (10 degrees apart!)
#lat and lon dependign on year. So that value is basically useless.
# for now just make site index with year and sort on unique values for those

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
dominantThresh = 0.75
dominantLocs = pd.DataFrame()
noGoodCnt = 0
for unSite in combdf['siteID'].unique(): 
    sitedf = combdf[combdf['siteID'] == unSite]
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
        else:
            BARat = sitedf['BALiveAc']/sitedf['BALiveAc'].sum()
            #check if one species is dominant in terms of basal area
            if BARat.max()>dominantThresh: 
                if len(sitedf[BARat>dominantThresh])>1:
                    raise Exception('Multiple dominant species are impossible')
                dominantLocs = pd.concat([dominantLocs, sitedf[BARat>dominantThresh]])
            else:
                noGoodCnt += 1
                
#with COND_CN, leads to 138, 172 dominantLocs
#of which there are 138,070 unique site IDs (so 100 plots within 4 km of each other)
#noGoodCnt = 275,815.
#with siteIndex as iterator above, have 57,844 dominantLocs, and 98,000 no good ones
#339 unique species codes
print( 'Minimum dominant lat: ' + str(np.min(dominantLocs['LAT'])) )
print( 'Maximum dominant lat: ' + str(np.max(dominantLocs['LAT'])) )
print( 'Minimum dominant lon: ' + str(np.min(dominantLocs['LON'])) )
print( 'Maximum dominant lon: ' + str(np.max(dominantLocs['LON'])) )

## Save to PWS grid
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

print( 'Minimum indX: ' + str(np.min(indX)) )
print( 'Maximum indX: ' + str(np.max(indX)) )
print( 'Minimum indY: ' + str(np.min(indY)) )
print( 'Maximum indY: ' + str(np.max(indY)) )
print('Shape pws: ')
pws.shape

#create array, and assign 
domSpec = np.zeros(pws.shape) * np.nan
#-1-D slice to assign to just a few points?
#for yVal in indY:
#    locNo = np.where(indY == yVal)
domSpec[indY, indX] = dominantLocs['SPCD']
valuedPoints = np.sum( ~np.isnan(domSpec) )
print('TEST number of non-zero grid points is: ', str(valuedPoints))
print('TEST hopefully equivalent # of indices is: ', str(len(indX)))
#sense-check. These should be the same to make sure original aggregation was correct
#answer: for dominantThresh 0.75: 33926 pix, doinatThresh = 0.85: 27340
#dominantThresh 0.8: 30658. Just leave at 0.75
#note that a large amount of these are not in relevant state sor where other vars have values

#pull out two basal area estimates
BALiveAcMap = np.zeros(pws.shape) * np.nan
BALiveAcMap[indY, indX] = dominantLocs['BALiveAc']
BALIVEMap = np.zeros(pws.shape) * np.nan
BALIVEMap[indY, indX] = dominantLocs['BALIVE']

#count how many pixels
BAMapCopy = BALiveAcMap.copy()
BAMapCopy[np.isnan(pws)] = np.nan
nDomFIAPix = np.sum(~np.isnan(BAMapCopy))
print('dominance threshold % BA at each site is : ' + str(dominantThresh) )
print('# of PWS pixels with FIA pllots is : ' + str(nDomFIAPix) )

#calculate how many FIA plots total across US
allFIAMap = np.zeros(pws.shape) * np.nan
allIndX = np.floor( (combdf['LON']-gt[0])/gt[1] ).to_numpy().astype(int)
allIndY = np.floor( (combdf['LAT']-gt[3])/gt[5] ).to_numpy().astype(int)
n1, n2 = pws.shape
#this has pixels in American Samoa and the like. Remove those
allMask = np.ones(allIndX.shape, dtype=bool)
allMask[allIndX < 0] = False
allMask[allIndX >= n2] = False
allMask[allIndY < 0] = False
allMask[allIndY > n1] = False
allIndX = allIndX[allMask]
allIndY = allIndY[allMask]
allFIAMap[allIndY, allIndX] = combdf['BALiveAc'][allMask]
nFIAPix = np.sum(~np.isnan(allFIAMap))
print('# of FIA plots in Western US is : ' + str(nFIAPix) )
error

#save geotiff
driver = gdal.GetDriverByName('GTiff')
output_file = 'FIADomSpecies.tif'
dataset = driver.Create(output_file, pws_x,pws_y, 1, gdal.GDT_Float32)
dataset.SetGeoTransform(gt)
dataset.GetRasterBand(1).WriteArray(domSpec)
dataset.FlushCache()#Writetodisk.
dataset=None

driver = gdal.GetDriverByName('GTiff')
output_file = 'FIABasalAreaAc.tif'
dataset = driver.Create(output_file, pws_x,pws_y, 1, gdal.GDT_Float32)
dataset.SetGeoTransform(gt)
dataset.GetRasterBand(1).WriteArray(BALiveAcMap)
dataset.FlushCache()#Writetodisk.
dataset=None

driver = gdal.GetDriverByName('GTiff')
output_file = 'FIABasalArea.tif'
dataset = driver.Create(output_file, pws_x,pws_y, 1, gdal.GDT_Float32)
dataset.SetGeoTransform(gt)
dataset.GetRasterBand(1).WriteArray(BALIVEMap)
dataset.FlushCache()#Writetodisk.
dataset=None



'''Now repeat with P50 file to make sure gets fewer points

# write function and make site ID
lonInd = np.floor(combdf['LON']*40)
latInd = np.floor(combdf['LAT']*40)
cheapSiteID = lonInd*1e5 + latInd 
p50df['siteID'] = cheapSiteID

#then try to find dominant locations in FIA_traits_Alex
p50Locs = pd.DataFrame()
noGoodP50Cnt = 0
troublePix = 0
for unSite in p50df['siteID'].unique(): 
    sitedf = p50df[p50df['siteID'] == unSite]
    if len(sitedf) == 1:
        p50Locs = pd.concat([p50Locs, sitedf])
    else:
        #if multiple years of meas at same site, pick the most recent
        if sitedf['measyr'].nunique() > 1:
            sitedf = sitedf[sitedf['measyr'] == sitedf['measyr'].max()]
        #note that if there's two sites within 4 km, they can have the same 
        #species across the two sites, and harder to check which is dominant
        #check for this and treat the whole set-up differently then
        if sitedf['FORTYPCD'].nunique() < len(sitedf):
            ##raise Exception('seems to be a repeated forest type')
            troublePix += 1
            continue
            ##give up on this for now, fixonly if decide to use this file.
        #check if one species is dominant in terms of basal area
        if sitedf['BA trait coverage'].max()>0.75: 
            #if len(sitedf['BA trait coverage']>0.75)>1:
            #    raise Exception('Multiple dominant species are impossible')
            p50Locs = pd.concat([p50Locs, sitedf[sitedf['BA trait coverage']>0.75] ])
        else:
            noGoodP50Cnt += 1
#indeed this is only about 25,000 locations, so don't use this
'''



