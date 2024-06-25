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
fn = 'MATCH/2023092712_RCTP.nc'
f = nc.Dataset(fn,'r')
print(f)
print(f.variables)
# %% ------------------------------------------------------------------------
# Put the 0 m data (T, P, RH) as the training data
Ts = f['T'][0][0]
Ps = f['P'][0][0]
RHs = f['RH'][0][0]
# %% ------------------------------------------------------------------------
times = f['times'][0]
H = f['H'][0]
T = f['T'][0].filled(fill_value=np.nan)
Qv = f['Qv'][0]
# %% ------------------------------------------------------------------------
plt.plot(T, H, label='temp')
plt.plot(Qv, H, label='vapor density')
plt.ylabel('height (m)')
plt.legend()
plt.show()
# %% ------------------------------------------------------------------------
f.close()