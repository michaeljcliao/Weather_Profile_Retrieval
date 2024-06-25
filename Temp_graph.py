import os
import numpy as np
import matplotlib.pyplot as plt
# %% ------------------------------------------------------------------------
os.chdir('/NFS15/michaelliao/retrievaldata')
# %% ------------------------------------------------------------------------
eval_y_T = np.loadtxt('./eval_y_Temp.csv', delimiter=',', dtype=np.float64)
eval_y_T_lv2 = np.loadtxt('./eval_y_Temp_lv2.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
os.chdir('./20240607_result')
eval_y_T_NN_5 = np.loadtxt('./eval_y_Temp_NN_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_RF_5 = np.loadtxt('./eval_y_Temp_RandomForest_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_XGB_5 = np.loadtxt('./eval_y_Temp_XGBoost_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
os.chdir('..')
os.chdir('./20240611_result')
eval_y_T_NN_18 = np.loadtxt('./eval_y_Temp_NN_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_RF_18 = np.loadtxt('./eval_y_Temp_RandomForest_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_XGB_18 = np.loadtxt('./eval_y_Temp_XGBoost_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_NN_27 = np.loadtxt('./eval_y_Temp_NN_pc27.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_RF_27 = np.loadtxt('./eval_y_Temp_RandomForest_pc27.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_XGB_27 = np.loadtxt('./eval_y_Temp_XGBoost_pc27.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_RF_1001 = np.loadtxt('./eval_y_Temp_RandomForest_1001.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_T_XGB_1001 = np.loadtxt('./eval_y_Temp_XGBoost_1001.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
os.chdir('..')
os.chdir('./20240613_result')
eval_y_T_NN_10 = np.loadtxt('./eval_y_Temp_NN_pc10.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
n_sample = 50
max_height = 1001
plt.plot(eval_y_T[n_sample], range(max_height), label = 'ground truth')
plt.plot(eval_y_T_lv2[n_sample], range(max_height), label = 'lvl2.nc old pred')
# plt.plot(eval_y_T_NN_5[n_sample], range(max_height), label = 'NN pred (5 PCs)')
# plt.plot(eval_y_T_RF_5[n_sample], range(max_height), label = 'RF pred (5 PCs)')
# plt.plot(eval_y_T_XGB_5[n_sample], range(max_height), label = 'XGB pred (5 PCs)')
plt.plot(eval_y_T_NN_10[n_sample], range(max_height), label = 'NN pred (10 PCs)')
plt.plot(eval_y_T_NN_18[n_sample], range(max_height), label = 'NN pred (18 PCs)')
plt.plot(eval_y_T_RF_18[n_sample], range(max_height), label = 'RF pred (18 PCs)')
plt.plot(eval_y_T_XGB_18[n_sample], range(max_height), label = 'XGB pred (18 PCs)')
# plt.plot(eval_y_T_NN_27[n_sample], range(max_height), label = 'NN pred (27 PCs)')
# plt.plot(eval_y_T_RF_27[n_sample], range(max_height), label = 'RF pred (27 PCs)')
# plt.plot(eval_y_T_XGB_27[n_sample], range(max_height), label = 'XGB pred (27 PCs)')
# plt.plot(eval_y_T_RF_1001[n_sample], range(max_height), label = 'RF pred (1001 dims)')
# plt.plot(eval_y_T_XGB_1001[n_sample], range(max_height), label = 'XGB pred (1001 dims)')

plt.legend()
plt.xlabel('Temperature (degree Celcius)')
plt.ylabel('Altitude (*10m)')
plt.title(f'Evaluation Dataset Sample Number %d' % n_sample)
plt.show()

# %% ------------------------------------------------------------------------
n_sample = 60
max_height = 1001
plt.plot(eval_y_T[n_sample], range(max_height), label = 'ground truth')
plt.plot(eval_y_T_lv2[n_sample], range(max_height), label = 'lvl2.nc old pred')
plt.plot(eval_y_T_NN_5[n_sample], range(max_height), label = 'NN pred (5 PCs)')
# plt.plot(eval_y_T_RF_5[n_sample], range(max_height), label = 'RF pred (5 PCs)')
# plt.plot(eval_y_T_XGB_5[n_sample], range(max_height), label = 'XGB pred (5 PCs)')
plt.plot(eval_y_T_NN_10[n_sample], range(max_height), label = 'NN pred (10 PCs)')
plt.plot(eval_y_T_NN_18[n_sample], range(max_height), label = 'NN pred (18 PCs)')
# plt.plot(eval_y_T_RF_18[n_sample], range(max_height), label = 'RF pred (18 PCs)')
# plt.plot(eval_y_T_XGB_18[n_sample], range(max_height), label = 'XGB pred (18 PCs)')
plt.plot(eval_y_T_NN_27[n_sample], range(max_height), label = 'NN pred (27 PCs)')
# plt.plot(eval_y_T_RF_27[n_sample], range(max_height), label = 'RF pred (27 PCs)')
# plt.plot(eval_y_T_XGB_27[n_sample], range(max_height), label = 'XGB pred (27 PCs)')
# plt.plot(eval_y_T_RF_1001[n_sample], range(max_height), label = 'RF pred (1001 dims)')
# plt.plot(eval_y_T_XGB_1001[n_sample], range(max_height), label = 'XGB pred (1001 dims)')

plt.legend()
plt.xlabel('Temperature (degrees Celsius)')
plt.ylabel('Altitude (*10m)')
plt.title(f'Evaluation Dataset Sample Number %d' % n_sample)
plt.show()
# %% ------------------------------------------------------------------------
os.chdir('..')
os.chdir('./20240613_result')
rmse_T_lv2 = np.loadtxt('./rmse_T_lv2.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_T_NN_5 = np.loadtxt('./rmse_T_NN_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_T_NN_10 = np.loadtxt('./rmse_T_NN_pc10.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_T_NN_18 = np.loadtxt('./rmse_T_NN_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_T_NN_27 = np.loadtxt('./rmse_T_NN_pc27.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_T_RF_5 = np.loadtxt('./rmse_T_RF_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_T_RF_18 = np.loadtxt('./rmse_T_RF_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_T_XGB_5 = np.loadtxt('./rmse_T_XGB_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_T_XGB_18 = np.loadtxt('./rmse_T_XGB_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
# Draw plot of rmse of different method on Temp
plt.figure(figsize = (5, 8))
max_height = 1001
plt.plot(rmse_T_lv2, range(max_height), label = 'lv2.nc old pred')
plt.plot(rmse_T_NN_5, range(max_height), label = 'NN pred (5 PCs)')
plt.plot(rmse_T_NN_10, range(max_height), label = 'NN pred (10 PCs)')
plt.plot(rmse_T_NN_18, range(max_height), label = 'NN pred (18 PC)')
plt.plot(rmse_T_NN_27, range(max_height), label = 'NN pred (27 PCs)')
# plt.plot(rmse_T_RF_5, range(max_height), label = 'RF pred (5 PCs)')
# plt.plot(rmse_T_RF_18, range(max_height), label = 'RF pred (18 PCs)')
# plt.plot(rmse_T_XGB_5, range(max_height), label = 'XGB pred (5 PCs)')
# plt.plot(rmse_T_XGB_18, range(max_height), label = 'XGB pred (18 PCs)')
plt.legend()
plt.xlabel('Difference Between Prediction and Truth (K)')
plt.ylabel('Altitude (*10 m)')
plt.title('RMSE of Temperature')
plt.show()

# %%
