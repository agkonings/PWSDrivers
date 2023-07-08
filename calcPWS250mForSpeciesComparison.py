# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:27:24 2023

@author: konings
"""

#load master

#find row/rat with master DFMC that is most specific
#mhave dominantLocs save HDF

#but want LFMC!

#to be efficient, create master similar to original PWS but do at point scale
#will have mismatch between resolution of DFMC and LFMC

#so load lats and lons from rf_regression/df_wSpec pickle

#current format is such that each master row is one point in time and one location
#keep it that way so can re-use 
#Krishan does it one day at a time
#here, repat, but instead of adding all locations add a loop where you pull