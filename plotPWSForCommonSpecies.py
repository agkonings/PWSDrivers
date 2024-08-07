# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:08:51 2022

@author: konings
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1, style = "ticks")
plt.rcParams.update({'font.size': 16})

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

'''
To pull out most common species in the Western US, need to filter first for
FIA sites (whether dominated by one species or not) that have PWS value
(as a proxy for being in the Western US domain). This means filtering stuff
outside the rectangualr bounds of the Western US matrix, but also outside of
Western Kansas and other adjoining states that are in the matrix but o tin the 
PWS domain
'''

#Load PWS lats and lons, geotransform
ds = gdal.Open('G:/My Drive/0000WorkComputer/dataStanford/PWSCalc/PWS_through2021_allSeas_nonorm_4monthslag_exact6years.tif')
gt = ds.GetGeoTransform()
pws = ds.GetRasterBand(1).ReadAsArray()
pws_y,pws_x = pws.shape
wkt_projection = ds.GetProjection()
ds = None

#calculate array indices into pws array for combdf locations
indX = np.floor( (combdf['LON']-gt[0])/gt[1] ).to_numpy().astype(int)
indY = np.floor( (combdf['LAT']-gt[3])/gt[5] ).to_numpy().astype(int)
#filter out of bounds locations, e.g. outside of rectangular grid
mask = np.ones(indX.shape, dtype=bool)
mask[indX < 0] = False
mask[indX >= pws_x] = False
mask[indY < 0] = False
mask[indY > pws_y] = False
indX = indX[mask]
indY = indY[mask]
combdf = combdf[mask]

#add pws value to combdf
pwsFromCombdf = pws[indY, indX]
combdf['PWS'] = pwsFromCombdf
combdf.dropna(inplace=True)

#Sort species by largest liveAc and pick 10 most common species
df_by_species = combdf.groupby(['SPCD']).sum()
topSpec = df_by_species.sort_values(by='BALiveAc', ascending=False).head(5)
#manually convert to species name for simplicity
#202 = Douglas-fir
#122 = ponderosa pine
#65 = Utah juniper
#263 = western hemlock
#108 = lodgepole pine

#or if take top five of jus tvalues count so don't weigh big species extra
combdf['SPCD'].value_counts()
#202 = Douglas-fir
#122 = ponderosa pine
#108 = lodgepole pine = 9201
#106 = common Pinyon (Pins edulis)
#65 = Utah juniper = 7422
#93 = Engelmann spruce
#263 = Western Hemlock = 
#19 = subalpine fir
#611 = sweetgum
#756 = honey mesquite = 5623
#15 = white fir = 5294
#17 = grand fir = 4684
#746 = quaking aspen = 4157
#827 = water oak = quercus nigra = 3986
#814 = Gambel oak = 3793
#242 = western redcedar = 3723
#807 = Blue oak = 174

#look at common species at locations where  asingle species dominates FIA plot
pickleLoc = './data/dominantLocs.pkl'
with open(pickleLoc, 'rb') as file:
    dominantLocs = pickle.load(file)
#five most common among dominatLocs
#131 = oblolly pine
#202 = Douglas-fir
#756 = honey mesquite
#122 = ponderosa pine
#65 = Utah juniper
#108 is sixth most common = lodgepole pine

#Missing common in dominantLocs
#108 = lodgepole pine = in dominantLocs
#106= common Pinyon = X in dominantLocs

#look at common species at actual sites used in study (also filtered for NLCD, data availability)
pickleLoc = './data/df_wSpec.pkl'
rfLocs = pd.read_pickle(pickleLoc)
vc = rfLocs['species'].value_counts()
#five most common among actual sites
#756 = honey mesquite
#65 = Utah juniper
#122 = ponderosa pine
#202 = Douglas-fir
#69 = oneseed juniper (juniperus monisperma)

'''
Ok, ready to plot!
'''

plotSpecList = [202, 756, 122, 65, 69]
legLabels = ["Douglas-fir", "Honey mesquite", "Ponderosa pine", "Utah juniper", "Oneseed juniper", "All"]

#filter dataframe to be used for plotting purposes so that it only 
#contains sites where the dominant cover is one of the species to be plotted
topDomLocs = rfLocs.copy()
noDataRows = topDomLocs.loc[~topDomLocs.species.isin(plotSpecList)]
topDomLocs.drop(noDataRows.index, inplace=True)


fig, ax = plt.subplots(figsize=(2, 2))
ax1 = sns.displot( topDomLocs, x='pws', hue='species', kind='kde', common_norm=False, \
            palette=sns.color_palette(n_colors=5), fill=False, bw_adjust=0.75, legend=False)
sns.kdeplot(rfLocs['pws'], ax=ax1, color='k', bw_adjust=0.75)
plt.ylabel("Density", size=18); plt.xticks(fontsize=16)
plt.xlabel("PWS", size=18); plt.yticks([], fontsize=16)
plt.xlim(0,6)
plt.legend(legLabels, loc="lower center", bbox_to_anchor=(0.5,-0.5), ncol=2, title=None, fontsize=18)
plt.savefig("../figures/PWSDriversPaper/PWSkdesbyspecies.eps", dpi=600)
plt.savefig("../figures/PWSDriversPaper/PWSkdesbyspecies.pdf", dpi=600)
plt.savefig("../figures/PWSDriversPaper/PWSkdesbyspecies.png", dpi=600)

'''
Calculate standard deviation of species means vs mean standard deviation per species
for use in paper text
'''
stdPerSpecList = []
mnPerSpecList = []
for spec in np.unique(rfLocs['species']):
    if vc[spec] > 50:
        thisMean = rfLocs[rfLocs['species'] == spec].pws.mean()
        thisStd = rfLocs[rfLocs['species'] == spec].pws.std()
        mnPerSpecList.append(thisStd)
        stdPerSpecList.append(thisStd)

print( 'Standard deviation across species means is ', str(np.std(mnPerSpecList)) )
print( 'Mean value of within-species standard deviations is ', str(np.mean(stdPerSpecList)) )

#plot bar chart of top species for paper
#don't overcomplicat ethings, just do manually
fig, ax = plt.subplots(figsize=(3,3))
ax = sns.barplot(y=[0,1,2,3,4], x=vc[plotSpecList], palette=sns.color_palette(n_colors=5), orient='h')
ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(legLabels[:-1]) 
ax.set(xlabel='number of sites\n(out of 21,455)')
plt.savefig("../figures/PWSDriversPaper/nSites_keySpecies.svg", dpi=300, bbox_inches='tight')
plt.show()