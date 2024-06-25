#!/usr/bin/env python3
# %% ------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from glob import glob

# %% ------------------------------------------------------------------------
os.chdir('/NFS14/schsu/202403-ChiaYi')
# %% ------------------------------------------------------------------------
random_seed = 7036
# %% ------------------------------------------------------------------------
# PROCESS X VARIABLES
fns = glob('MATCH/*_lv1.nc')
fns.sort()
print(len(fns))
# %% ------------------------------------------------------------------------
# stick all x variables together into one 1d array, skipping the Tb data
# from the 24GHz channel
# Rain, FLGs, FLGTb are excluded due to their binary property
# Tir is excluded due to the irrelavence of IRT
fn = fns[0]
f = nc.Dataset(fn,'r')
VarX = np.concatenate((f['Tb'][0][0:7],
                       f['Tb'][0][8:],
                       f['Ts'][:],
                       f['Ps'][:],
                       f['RHs'][:],
                       f['Rain'][:]))
VarX = np.reshape(VarX, (1, -1))

for fn in fns[1:]:
  f = nc.Dataset(fn,'r')
  VarX_new = np.concatenate((f['Tb'][0][0:7],
                             f['Tb'][0][8:],
                             f['Ts'][:],
                             f['Ps'][:],
                             f['RHs'][:],
                             f['Rain'][:]))
  VarX_new = np.reshape(VarX_new, (1, -1))
  VarX = np.concatenate((VarX, VarX_new), axis=0)

# %% ------------------------------------------------------------------------
# Subset the x values to match the y values
# VarX_train = VarX[~np.isnan(T).any(axis=1)]
# VarX_test = VarX[np.isnan(T).any(axis=1)]
# * Subset the x values with Rain == 0 (sunny) to be kept for training'
# rainy data will not be used
VarX_sunny = VarX[VarX[:, -1] == 0]
VarX_sunny = VarX_sunny[:, :-1] # remove the final column after subsetting
# %% ------------------------------------------------------------------------
# PROCESS Y VARIABLES
# Find all RCTP files, pick up the T and Qv
fns = glob('MATCH/*_RCTP.nc')
fns.sort()
print(len(fns))
with nc.MFDataset(fns,'r') as fs:
  T = fs['T'][:].filled(fill_value=np.nan)
  Qv = fs['Qv'][:].filled(fill_value=np.nan)
  print(T.shape)
  print(Qv.shape)

# %% ------------------------------------------------------------------------
# Hand pick RS41 and ST, select RS41 file only if both files exist
# RS41 files are missing Td, making nc.MFDataset() not working
fileRS41nST = ['MATCH/2024031515_ST.nc', 'MATCH/2024031518_ST.nc',
               'MATCH/2024031521_ST.nc', 'MATCH/2024031600_RS41.nc',
               'MATCH/2024031604_ST.nc', 'MATCH/2024031606_ST.nc',
               'MATCH/2024031609_ST.nc', 'MATCH/2024031612_RS41.nc',
               'MATCH/2024031615_ST.nc', 'MATCH/2024031618_ST.nc',
               'MATCH/2024031621_ST.nc', 'MATCH/2024031700_RS41.nc',
               'MATCH/2024031706_ST.nc', 'MATCH/2024031709_ST.nc',
               'MATCH/2024031711_ST.nc', 'MATCH/2024031712_RS41.nc',
               'MATCH/2024031715_ST.nc', 'MATCH/2024031718_ST.nc',
               'MATCH/2024031721_RS41.nc', 'MATCH/2024031723_ST.nc',]

for file in fileRS41nST:
  fn = file
  f = nc.Dataset(fn,'r')
  T_file = f['T'][:].filled(fill_value=np.nan)
  Qv_file = f['Qv'][:].filled(fill_value=np.nan)
  # print(T_file.shape)
  # print(Qv_file.shape)
  T = np.concatenate((T, T_file), axis=0)
  Qv = np.concatenate((Qv, Qv_file), axis=0)

# %% ------------------------------------------------------------------------
# * Pick only rows that corresponds to sunny lvl1 data
T_sunny = T[VarX[:, -1] == 0]
Qv_sunny = Qv[VarX[:, -1] == 0]
print(T_sunny.shape)
print(Qv_sunny.shape)

# %% ------------------------------------------------------------------------
# The rows with missing values are the same for T and Qv (The ones where the
# balloon couldn't fly as high)
print((np.isnan(T_sunny).any(axis=1)==np.isnan(Qv_sunny).any(axis=1)))

# %%
# Pick rows with no missing values as the y value of the training set
T_train = T_sunny[~np.isnan(T_sunny).any(axis=1)]
Qv_train = Qv_sunny[~np.isnan(T_sunny).any(axis=1)]

# Pick rows with missing values as the test set
T_test = T_sunny[np.isnan(T_sunny).any(axis=1)]
Qv_test = Qv_sunny[np.isnan(T_sunny).any(axis=1)]
print(T_train.shape)
print(Qv_train.shape)
print(T_test.shape)
print(Qv_test.shape)
# %% ------------------------------------------------------------------------
# triple the training data
T_train_copy = T_train.copy()
Qv_train_copy = Qv_train.copy()
T_train_augmented = np.concatenate(
  (T_train, T_train_copy, T_train_copy), axis=0)
Qv_train_augmented = np.concatenate(
  (Qv_train, Qv_train_copy, Qv_train_copy), axis=0)
print(T_train_augmented.shape)
print(Qv_train_augmented.shape)
# %% ------------------------------------------------------------------------
# subset the X data according to Y data
VarX_train = VarX_sunny[~np.isnan(T_sunny).any(axis=1)]
VarX_test = VarX_sunny[np.isnan(T_sunny).any(axis=1)]
shape_train_x = VarX_train.shape
VarX_train_org = VarX_train.copy()
# %% ------------------------------------------------------------------------
# Add one N(0,1) Noise
np.random.seed(random_seed)
mu, sigma = 0, 1
X_noise_train = np.random.normal(mu, sigma, shape_train_x)
VarX_train_aug = VarX_train_org + X_noise_train
VarX_train = np.concatenate((VarX_train, VarX_train_aug), axis=0)
# %% ------------------------------------------------------------------------
# Subtract one N(0,0.8) Noise
np.random.seed(random_seed)
mu, sigma = 0, 0.8
X_noise_train = np.random.normal(mu, sigma, shape_train_x)
VarX_train_aug = VarX_train_org - X_noise_train
VarX_train = np.concatenate((VarX_train, VarX_train_aug), axis=0)
# * THERE ARE TRIPLE DATA NOW
# %% ------------------------------------------------------------------------
# Get the lvl2 data, those created by previous ML model
fns = glob('MATCH/*_lv2.nc')
fns.sort()
fn = fns[0]
f = nc.Dataset(fn,'r')
T_lv2 = f['T'][:].filled(fill_value=np.nan)
Qv_lv2 = f['Qv'][:].filled(fill_value=np.nan)
for fn in fns[1:]:
  f = nc.Dataset(fn,'r')
  T_lv2 = np.concatenate((T_lv2, f['T'][:].filled(fill_value=np.nan)), axis=0)
  Qv_lv2 = np.concatenate((Qv_lv2, f['Qv'][:].filled(fill_value=np.nan)), axis=0)
print(T_lv2.shape)
print(Qv_lv2.shape)
# %% ------------------------------------------------------------------------
# subset the sunny data, then split into training and test
T_lv2_sunny = T_lv2[VarX[:, -1] == 0]
Qv_lv2_sunny = Qv_lv2[VarX[:, -1] == 0]
T_lv2_train = T_lv2_sunny[~np.isnan(T_sunny).any(axis=1)]
Qv_lv2_train = Qv_lv2_sunny[~np.isnan(Qv_sunny).any(axis=1)]
# %% ------------------------------------------------------------------------
# triple the data
T_lv2_train_copy = T_lv2_train.copy()
Qv_lv2_train_copy = Qv_lv2_train.copy()
T_lv2_train_augmented = np.concatenate(
  (T_train, T_lv2_train_copy, T_lv2_train_copy), axis=0)
Qv_lv2_train_augmented = np.concatenate(
  (Qv_train, Qv_lv2_train_copy, Qv_lv2_train_copy), axis=0)
T_lv2_test = T_lv2_sunny[np.isnan(T_sunny).any(axis=1)]
Qv_lv2_test = Qv_lv2_sunny[np.isnan(Qv_sunny).any(axis=1)]

# %% ------------------------------------------------------------------------
# change the directory back to store data as csv
os.chdir('/NFS15/michaelliao/retrievaldata')

# %% ------------------------------------------------------------------------
np.savetxt("train_x.csv", VarX_train, delimiter=",")
np.savetxt("train_y_Temp.csv", T_train_augmented, delimiter=",")
np.savetxt("train_y_Qv.csv", Qv_train_augmented, delimiter=",")
np.savetxt("eval_x.csv", VarX_test, delimiter=",")
np.savetxt("eval_y_Temp.csv", T_test, delimiter=",")
np.savetxt("eval_y_Qv.csv", Qv_test, delimiter=",")

np.savetxt("train_y_Temp_lv2.csv", T_lv2_train_augmented, delimiter=",")
np.savetxt("train_y_Qv_lv2.csv", Qv_lv2_train_augmented, delimiter=",")
np.savetxt("eval_y_Temp_lv2.csv", T_lv2_test, delimiter=",")
np.savetxt("eval_y_Qv_lv2.csv", Qv_lv2_test, delimiter=",")

# Save original y dataset
np.savetxt("train_y_Temp_original.csv", T_train, delimiter=",")
np.savetxt("train_y_Qv_original.csv", Qv_train, delimiter=",")
np.savetxt("train_y_Temp_lv2_original.csv", T_lv2_train, delimiter=",")
np.savetxt("train_y_Qv_lv2_original.csv", Qv_lv2_train, delimiter=",")
# %%