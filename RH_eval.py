import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# %% ------------------------------------------------------------------------
os.chdir('/NFS15/michaelliao/retrievaldata')
# %% ------------------------------------------------------------------------
eval_y_RH = np.loadtxt('./eval_y_RH.csv', delimiter=',', dtype=np.float64)
eval_y_RH_lv2 = np.loadtxt('./eval_y_RH_lv2.csv',
                          delimiter=',',
                          dtype=np.float64)
train_y_RH_lv2 = np.loadtxt('./train_y_RH_lv2.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
# Get Temp prediction from eval_x
os.chdir('./20240613_result')
eval_y_T_NN_10 = np.loadtxt('./eval_y_Temp_NN_pc10.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
# Get Qv prediction from eval_x
os.chdir('..')
os.chdir('./20240611_result')
eval_y_Qv_NN_18 = np.loadtxt('./eval_y_Qv_NN_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
# Derive relative humidity for the eval dataset
eval_saturatedVaporPrPow = (7.5*eval_y_T_NN_10)/(237.3+eval_y_T_NN_10) # T: Celsius
eval_y_T_NN_10 = 273.15 + eval_y_T_NN_10 # convert to Kelvin
eval_actualVaporPr = eval_y_Qv_NN_18/1000*eval_y_T_NN_10*461.5 /100
eval_saturatedVaporPr = 6.11 * 10**eval_saturatedVaporPrPow

eval_y_RH_pred = eval_actualVaporPr/eval_saturatedVaporPr * 100
# %% ------------------------------------------------------------------------
n_sample = 20
plt.plot(eval_y_RH[n_sample], range(1001), label = 'ground truth')
plt.plot(eval_y_RH_lv2[n_sample], range(1001), label = 'lvl2.nc old pred')
plt.plot(eval_y_RH_pred[n_sample], range(1001), label = 'prediction')
plt.legend()
plt.title(f'Evaluation Dataset Sample Number %d' % n_sample)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Altitude (*10 m)')
plt.show()
# %% ------------------------------------------------------------------------
n_sample = 36
plt.plot(eval_y_RH[n_sample], range(1001), label = 'ground truth')
plt.plot(eval_y_RH_lv2[n_sample], range(1001), label = 'lvl2.nc old pred')
plt.plot(eval_y_RH_pred[n_sample], range(1001), label = 'prediction')
plt.legend()
plt.title(f'Evaluation Dataset Sample Number %d' % n_sample)
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Altitude (*10 m)')
plt.show()
# %% ------------------------------------------------------------------------
# ! RMSE not reliable, should not use eval data
# Create (prediction - truth) / truth percentage against altitude
eval_y_RH_diff = eval_y_RH_pred - eval_y_RH
# eval_y_RH_sqrerr = eval_y_RH_diff/eval_y_RH
eval_y_RH_sqrerr = eval_y_RH_diff*eval_y_RH_diff
eval_y_RH_rmse = np.sqrt(np.mean(eval_y_RH_sqrerr, axis=0))
# %% ------------------------------------------------------------------------
# plt.figure(figsize = (5, 8))
plt.plot(eval_y_RH_rmse, range(1001))
plt.xlabel('Difference Between Prediction and Truth (%)')
plt.ylabel('Altitude (*10 m)')
plt.title('RMSE of Relative Humidity')
plt.show()

# %% ------------------------------------------------------------------------
# Get test prediction
os.chdir('..')
os.chdir('./20240617_result')
test_y_T_NN_10 = np.loadtxt('./test_pred_T_NN_pc10.csv',
                          delimiter=',',
                          dtype=np.float64)
test_y_Qv_NN_18 = np.loadtxt('./test_pred_Qv_NN_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
os.chdir('..')
train_x_csv = np.loadtxt('./train_x.csv', delimiter=',', dtype=np.float64)
train_y_RH_csv = np.loadtxt('./train_y_RH.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
# Derive relative humidity for the test dataset
test_saturatedVaporPrPow = (7.5*test_y_T_NN_10)/(237.3+test_y_T_NN_10) # T: Celsius
test_y_T_NN_10 = 273.15 + test_y_T_NN_10 # convert to Kelvin
test_actualVaporPr = test_y_Qv_NN_18/1000*test_y_T_NN_10*461.5 /100
test_saturatedVaporPr = 6.11 * 10**test_saturatedVaporPrPow

test_y_RH_pred = test_actualVaporPr/test_saturatedVaporPr * 100
# %% ------------------------------------------------------------------------
# Process the ground truth RH; only RH is interested
random_seed = 7036
_, _, train_y_RH, test_y_RH = train_test_split(train_x_csv, 
                                               train_y_RH_csv,
                                               test_size=0.2,
                                               random_state=random_seed)
# %% ------------------------------------------------------------------------
# Create (prediction - truth) against altitude
test_y_RH_diff = test_y_RH_pred - test_y_RH
test_y_RH_sqrerr = test_y_RH_diff*test_y_RH_diff
test_y_RH_rmse = np.sqrt(np.mean(test_y_RH_sqrerr, axis=0)) # average over the row
# %% ------------------------------------------------------------------------
# Split lv2 RH
_, _, train_y_lv2_RH, test_y_lv2_RH = train_test_split(train_x_csv, 
                                               train_y_RH_lv2,
                                               test_size=0.2,
                                               random_state=random_seed)
# %% ------------------------------------------------------------------------
# Create (prediction - truth) against altitude for lvl2 old prediction
test_y_lv2_RH_diff = test_y_lv2_RH - test_y_RH
test_y_lv2_RH_sqrerr = test_y_lv2_RH_diff*test_y_lv2_RH_diff
test_y_lv2_RH_rmse = np.sqrt(np.mean(test_y_lv2_RH_sqrerr, axis=0)) # average over the row
# %% ------------------------------------------------------------------------
plt.figure(figsize = (5, 8))
plt.plot(test_y_RH_rmse[0:200], range(200), label='Temp(NN 10 PCs) / Qv(NN 18 PCs)')
plt.plot(test_y_lv2_RH_rmse[0:200], range(200), label='lv2 old pred')
plt.legend()
plt.xlabel('Difference Between Prediction and Truth (%)')
plt.ylabel('Altitude (*10 m)')
plt.title('RMSE of Relative Humidity (Below 2000m)')
plt.show()
# %%
