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
fn = 'MATCH/2024031612_ST.nc'
f = nc.Dataset(fn,'r')
print(f)
print(f.variables)
# %% ------------------------------------------------------------------------
times = f['times'][0]
H = f['H'][0]
T = f['T'][0].filled(fill_value=np.nan)
Qv = f['Qv'][0]
print(H)
# %% ------------------------------------------------------------------------
plt.plot(T, H, label='temp')
plt.plot(Qv, H, label='vapor density')
plt.ylabel('height (m)')
plt.legend()
plt.show()
# %% ------------------------------------------------------------------------
f.close()