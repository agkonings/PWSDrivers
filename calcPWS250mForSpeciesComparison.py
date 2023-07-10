# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:27:24 2023

@author: konings
"""

import pandas 
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

dir_4km = "G:/My Drive/0000WorkComputer/dataStanford/"
dir_250m =''''.....ON OTHER DIRVE'''

'''
Actually now create master file to do the regression on, following 
the logic and code in wildfire_from_lfmc/analysis/pws_calculation
'''

#Find x_index_4km, y_index_4km values for FIA plots being studied

#Find x_index_250m, y_index_250m values for FIA plots being studied


#create master with all DFMC loaded but without LFMC
master = pd.DataFrame()
for date in dates:
    df = pd.DataFrame()
    filename = os.path.join(dir_4km, folder,"lfmc_map_%s.tif"%date)
    ds4km = gdal.Open(filename)    
    #dfmc: raster band 2 is for 100 hr rather than 1000 hr fuels
    dfmc = np.array(ds4km.GetRasterBand(2).ReadAsArray())
    ds4km = None    
    filename250m = os.path.join(dir_250m, folder,"lfmc_map_%s.tif"%date)
    ds250m = gdal.Open(filename)    
    lfmc = np.array(ds250m.GetRasterBand(1).ReadAsArray())
    ds250m = None
        
    df['dfmc(t)'] = dfmc[x_index_4km, y_index_4km]
    df['lfmc(t)'] = lfmc[x_index_250m, y_index_250m]
    df['pixel_index'] = df.index
    df['lat'] = FIA_wSpec['LAT']
    df['lon'] = FIA_wSpec['LON']
        
    df['date'] = date
    ctr = 1
    sys.stdout.write('\r'+'[INFO] Time step %s'%date)
    sys.stdout.flush()
    # print(date)
    subsetDates = get_dates(date, maxLag = maxLag)
    for t in subsetDates:
        shiftedFile = os.path.join(dir_data,folder,"lfmc_map_%s.tif"%t)
        ds = gdal.Open(shiftedFile)
        df['dfmc(t-%d)'%ctr] = np.array(ds.GetRasterBand(dfmcDict[hr]).ReadAsArray())[x_index_4km, y_index_4km]
        ctr+=1
    df.dropna(inplace = True)
    master = master.append(df,ignore_index = True)     
master = master.dropna()


#later group by pixel_index for regression