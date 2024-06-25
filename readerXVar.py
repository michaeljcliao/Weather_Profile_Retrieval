#!/usr/bin/env python3
# %%
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from glob import glob
# %% ------------------------------------------------------------------------
os.chdir('/NFS14/schsu/202403-ChiaYi')
# %% ------------------------------------------------------------------------
fns = glob('MATCH/*_lv1.nc')
fns.sort()
print(len(fns))
# %% ------------------------------------------------------------------------
# stick all x variables together into one 1d array, skipping the Tb data
# from the 24GHz channel
fn = fns[0]
f = nc.Dataset(fn,'r')
VarX = np.concatenate((f['Tb'][0][0:7],
                       f['Tb'][0][8:],
                       f['Ts'][:],
                       f['Ps'][:],
                       f['Tir'][:],
                       f['RHs'][:],
                       f['Rain'][:],
                       f['FLGs'][:],
                       f['FLGTb'][:]))
VarX = np.reshape(VarX, (1, -1))
for fn in fns[1:]:
  f = nc.Dataset(fn,'r')
  VarX_new = np.concatenate((f['Tb'][0][0:7],
                             f['Tb'][0][8:],
                             f['Ts'][:],
                             f['Ps'][:],
                             f['Tir'][:],
                             f['RHs'][:],
                             f['Rain'][:],
                             f['FLGs'][:],
                             f['FLGTb'][:]))
  VarX_new = np.reshape(VarX_new, (1, -1))
  VarX = np.concatenate((VarX, VarX_new), axis=0)
# %% ------------------------------------------------------------------------
print(VarX.shape)
# %% ------------------------------------------------------------------------
df_X_RCTP = pd.DataFrame(VarX[0:114])
df_X_rest = pd.DataFrame(VarX[114:])
# %% ------------------------------------------------------------------------
print(df_X_RCTP.to_markdown())
# %% ------------------------------------------------------------------------
f.close()