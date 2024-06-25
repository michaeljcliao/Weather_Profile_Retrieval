#!/usr/bin/env python3
# %%
import os,sys
import numpy as np
import netCDF4 as nc
from glob import glob
import matplotlib.pyplot as plt
# %% ------------------------------------------------------------------------
os.chdir('/NFS14/schsu/202403-ChiaYi')
# %% ------------------------------------------------------------------------
fn = 'MATCH/2023092300_lv1.nc'
f = nc.Dataset(fn,'r')
# print(f)
print(f.variables)
# %% ------------------------------------------------------------------------
times = f['times'][:]
freq = f['frequencies'][:] # get rid of 24GHz, the 8-th entry
Ts = f['Ts'][:]
Ps = f['Ps'][:]
Tir = f['Tir'][:]
RHs = f['RHs'][:]
Rain = f['Rain'][:]
FLGs = f['FLGs'][:]
Tb = f['Tb'][0][:]
# Tb = f['Tb'][0][0:7] # get rid of 24GHz, the 8-th entry
# Tb = np.concatenate([Tb, f['Tb'][0][8:]])
FLGTb = f['FLGTb'][:]
# %% ----------------------------------------------------------------
VarX = np.concatenate((f['Tb'][0][0:7],
                       f['Tb'][0][8:],
                       f['Ts'][:],
                       f['Ps'][:],
                       f['Tir'][:],
                       f['RHs'][:],
                       f['Rain'][:],
                       f['FLGs'][:],
                       f['FLGTb'][:]))
# %% ----------------------------------------------------------------
print(freq)
print(Ts)
print(Ps)
print(Tir)
print(RHs)
print(Rain)
print(FLGs)
print(Tb)
print(FLGTb)
print(VarX)
# %% 
print(VarX.shape)
# %% ----------------------------------------------------------------
f.close()