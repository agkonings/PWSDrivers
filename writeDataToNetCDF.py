# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:45:49 2024

@author: konings
"""

import xarray as xr
import pandas as pd
import numpy as np
import dill


#load dill session
#dill.load_session('C:/repos/data/RFregression_dill_backup_beforeRound2.pkl')
dill.load_session('C:/repos/data/rf_regression_dill.pkl')

'''
create dataframe with data-derived PWS, all drivers, and then as additional info
land cover, species, lat, lon, and the predictions from the RF and speices-based model
all except the predictions from teh two models are already in df_wSpec
'''
dfPWS = df_wSpec.copy()
dfPWS['predictions_from_species'] = pwsPred
dfPWS['predictions_from_RF'] = rfPredAll

# convert dataframe to xarray Dataset
ds = xr.Dataset.from_dataframe(dfPWS)
ds.attrs['description'] = 'Supporting data for "Tree species explain only half of explained spatial variability in plant water sensitivity"'
ds.attrs['author'] = 'Alexandra Konings (konings@stanford.edu)'

ds.to_netcdf('C:/repos/data/PWS_species.nc')
