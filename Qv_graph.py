import os
import numpy as np
import matplotlib.pyplot as plt
# %% ------------------------------------------------------------------------
os.chdir('/NFS15/michaelliao/retrievaldata')
# %% ------------------------------------------------------------------------
eval_y_Qv = np.loadtxt('./eval_y_Qv.csv', delimiter=',', dtype=np.float64)
eval_y_Qv_lv2 = np.loadtxt('./eval_y_Qv_lv2.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
os.chdir('./20240611_result')
eval_y_Qv_NN_5 = np.loadtxt('./eval_y_Qv_NN_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_Qv_RF_5 = np.loadtxt('./eval_y_Qv_RandomForest_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_Qv_XGB_5 = np.loadtxt('./eval_y_Qv_XGBoost_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_Qv_NN_10 = np.loadtxt('./eval_y_Qv_NN_pc10.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_Qv_NN_18 = np.loadtxt('./eval_y_Qv_NN_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_Qv_RF_18 = np.loadtxt('./eval_y_Qv_RandomForest_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
eval_y_Qv_XGB_18 = np.loadtxt('./eval_y_Qv_XGBoost_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
# eval_y_Qv_RF_1001 = np.loadtxt('./eval_y_Qv_RandomForest_1001.csv',
#                           delimiter=',',
#                           dtype=np.float64)
# eval_y_Qv_XGB_1001 = np.loadtxt('./eval_y_Qv_XGBoost_1001.csv',
#                           delimiter=',',
#                           dtype=np.float64)

# %% ------------------------------------------------------------------------
n_sample = 50
plt.plot(eval_y_Qv[n_sample], range(1001), label = 'ground truth')
# plt.plot(eval_y_Qv_lv2[n_sample], range(1001), label = 'lvl2.nc old pred')
plt.plot(eval_y_Qv_NN_5[n_sample], range(1001), label = 'NN pred (5 PCs)')
# plt.plot(eval_y_Qv_RF_5[n_sample], range(1001), label = 'RF pred (5 PCs)')
# plt.plot(eval_y_Qv_XGB_5[n_sample], range(1001), label = 'XGB pred (5 PCs)')
plt.plot(eval_y_Qv_NN_10[n_sample], range(1001), label = 'NN pred (10 PCs)')
plt.plot(eval_y_Qv_NN_18[n_sample], range(1001), label = 'NN pred (18 PCs)')
# plt.plot(eval_y_Qv_RF_18[n_sample], range(1001), label = 'RF pred (18 PCs)')
# plt.plot(eval_y_Qv_XGB_18[n_sample], range(1001), label = 'XGB pred (18 PCs)')
# plt.plot(eval_y_Qv_RF_1001[n_sample], range(1001), label = 'RF pred (1001 dims)')
# plt.plot(eval_y_Qv_XGB_1001[n_sample], range(1001), label = 'XGB pred (1001 dims)')

plt.legend()
plt.title(f'Evaluation Dataset Sample Number %d' % n_sample)
plt.xlabel('Vapor Density (g/cubic meter)')
plt.ylabel('Altitude (*10 m)')
plt.show()

# %% ------------------------------------------------------------------------
n_sample = 40
plt.plot(eval_y_Qv[n_sample], range(1001), label = 'ground truth')
# plt.plot(eval_y_Qv_lv2[n_sample], range(1001), label = 'lvl2.nc old pred')
# plt.plot(eval_y_Qv_NN_5[n_sample], range(1001), label = 'NN pred (5 PCs)')
# plt.plot(eval_y_Qv_RF_5[n_sample], range(1001), label = 'RF pred (5 PCs)')
# plt.plot(eval_y_Qv_XGB_5[n_sample], range(1001), label = 'XGB pred (5 PCs)')
# plt.plot(eval_y_Qv_NN_10[n_sample], range(1001), label = 'NN pred (10 PCs)')
plt.plot(eval_y_Qv_NN_18[n_sample], range(1001), label = 'NN pred (18 PCs)')
# plt.plot(eval_y_Qv_RF_18[n_sample], range(1001), label = 'RF pred (18 PCs)')
# plt.plot(eval_y_Qv_XGB_18[n_sample], range(1001), label = 'XGB pred (18 PCs)')
# plt.plot(eval_y_Qv_RF_1001[n_sample], range(1001), alabel = 'RF pred (1001 dims)')
# plt.plot(eval_y_Qv_XGB_1001[n_sample], range(1001), label = 'XGB pred (1001 dims)')


plt.legend()
plt.title(f'Evaluation Dataset Sample Number %d' % n_sample)
plt.xlabel('Vapor Density (g/cubic meter)')
plt.ylabel('Altitude (*10 m)')
plt.show()
# %% ------------------------------------------------------------------------
os.chdir('..')
os.chdir('./20240613_result')
rmse_Qv_lv2 = np.loadtxt('./rmse_Qv_lv2.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_Qv_NN_5 = np.loadtxt('./rmse_Qv_NN_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_Qv_NN_10 = np.loadtxt('./rmse_Qv_NN_pc10.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_Qv_NN_18 = np.loadtxt('./rmse_Qv_NN_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_Qv_NN_27 = np.loadtxt('./rmse_Qv_NN_pc27.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_Qv_RF_5 = np.loadtxt('./rmse_Qv_RF_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_Qv_RF_18 = np.loadtxt('./rmse_Qv_RF_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_Qv_XGB_5 = np.loadtxt('./rmse_Qv_XGB_pc5.csv',
                          delimiter=',',
                          dtype=np.float64)
rmse_Qv_XGB_18 = np.loadtxt('./rmse_Qv_XGB_pc18.csv',
                          delimiter=',',
                          dtype=np.float64)
# %% ------------------------------------------------------------------------
# Draw plot of rmse of different method on Qv
# plt.figure(figsize = (5, 8))
max_height = 150
plt.plot(rmse_Qv_lv2[0:max_height], range(max_height), label = 'lv2.nc old pred')
plt.plot(rmse_Qv_NN_5[0:max_height], range(max_height), label = 'NN pred (5 PCs)')
plt.plot(rmse_Qv_NN_10[0:max_height], range(max_height), label = 'NN pred (10 PCs)')
plt.plot(rmse_Qv_NN_18[0:max_height], range(max_height), label = 'NN pred (18 PCs)')
plt.plot(rmse_Qv_NN_27[0:max_height], range(max_height), label = 'NN pred (27 PCs)')
# plt.plot(rmse_Qv_RF_5[0:max_height], range(max_height), label = 'RF pred (5 PCs)')
plt.plot(rmse_Qv_RF_18[0:max_height], range(max_height), label = 'RF pred (18 PCs)')
# plt.plot(rmse_Qv_XGB_5[0:max_height], range(max_height), label = 'XGB pred (5 PCs)')
plt.plot(rmse_Qv_XGB_18[0:max_height], range(max_height), label = 'XGB pred (18 PCs)')
plt.legend()
plt.xlabel('Difference Between Prediction and Truth (Ratio)')
plt.ylabel('Altitude (*10 m)')
plt.title('RMSE of Vapor Density (1500m and below)')
plt.show()

# %%
