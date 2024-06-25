
# Author: Je-Ching Liao | Academia Sinica | University of Michigan
# Date: June 2024
# File Purpose: .py file to apply sklearn RF and XGBoosting for temperature and
#               vapor density retrieval from spectrometer
# %% ------------------------------------------------------------------------
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import time
# %% ------------------------------------------------------------------------
os.chdir('/NFS15/michaelliao/retrievaldata')
# %% ------------------------------------------------------------------------
random_seed = 7036
# %% ------------------------------------------------------------------------
train_x = np.loadtxt('./train_x.csv', delimiter=',', dtype=np.float64)
train_y_T = np.loadtxt('./train_y_Temp.csv', delimiter=',', dtype=np.float64)
print(train_x.shape)
print(train_y_T.shape)

# %% ------------------------------------------------------------------------

# %% ------------------------------------------------------------------------
# %% ------------------------------------------------------------------------
# our model: one output is predicted by one base estimator
# model = MultiOutputRegressor(base_estimator)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                             random_state=random_seed, 
                             criterion='squared_error',
                             max_depth=None, 
                             min_samples_split=2))

# %% ------------------------------------------------------------------------
# data split and train
X_train, X_test, Y_T_train, Y_T_test = train_test_split(train_x, 
                                                        train_y_T,
                                                        test_size=0.2,
                                                        random_state=random_seed)
print(Y_T_train.shape)
# %% ------------------------------------------------------------------------
# standardize the input
stdScaler = StandardScaler()
X_train = stdScaler.fit_transform(X_train)
X_test = stdScaler.transform(X_test)
# %% ------------------------------------------------------------------------
# PCA 
pca = PCA(n_components=18, random_state=random_seed)
# %% ------------------------------------------------------------------------
# reduce the dimensionality
Y_T_train_PCA = pca.fit_transform(Y_T_train)
Y_T_test_PCA = pca.transform(Y_T_test)
# %% ------------------------------------------------------------------------
# Check the percentage explained by components
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# %% ------------------------------------------------------------------------
start_time = time.time()
model.fit(X_train, Y_T_train_PCA)
# model.fit(X_train, Y_T_train)
end_time = time.time()
print(end_time - start_time)

# 269.69056010246277 (no PCA)
# 5.194640874862671 (18 PCs)
# 7.81648 (27 PCs)
# %% ------------------------------------------------------------------------
# test
Y_T_pred = model.predict(X_test)
Y_T_pred = pca.inverse_transform(Y_T_pred) # For pca'd results
# evaluate predictions
Y_T_train_pred = model.predict(X_train)
Y_T_train_pred = pca.inverse_transform(Y_T_train_pred) # For pca'd results
# %% ------------------------------------------------------------------------
mse = mean_squared_error(Y_T_train, Y_T_train_pred)
print(f'Training MSE =', mse)
mse = mean_squared_error(Y_T_test, Y_T_pred)
print(f'Test MSE =', mse)

# Training MSE = 0.2440861169757197; Test MSE = 1.738114029926516 (no PCA)
# Training MSE = 0.28054; Test MSE = 1.8659 (18 PCs)
# Training MSE = 0.2488947; Test MSE = 1.8395 (27 PCs)
# After correct Ts, Ps, RHs
# Training MSE = 0.68802; Test MSE = 2.11536 (5 PCs)
# Training MSE = 0.25959; Test MSE = 1.83167 (18 PCs)
# %% ------------------------------------------------------------------------
# Use eval data set
eval_x = np.loadtxt('./eval_x.csv', delimiter=',', dtype=np.float64)
eval_x = stdScaler.transform(eval_x)

eval_y_T_pred = model.predict(eval_x)
eval_y_T_pred = pca.inverse_transform(eval_y_T_pred) # For pca'd results
# %% ------------------------------------------------------------------------
# Save the result of RandomForestRegressor
np.savetxt("eval_y_Temp_RandomForest_pc18.csv", eval_y_T_pred, delimiter=",")
# %% ------------------------------------------------------------------------
# Create (prediction - truth) against altitude
Y_T_test_diff = Y_T_pred - Y_T_test
Y_T_test_sqrerr = Y_T_test_diff*Y_T_test_diff
Y_T_test_rmse = np.sqrt(np.mean(Y_T_test_sqrerr, axis=0)) # average over the row
# %% ------------------------------------------------------------------------
np.savetxt("rmse_T_RF_pc18.csv", Y_T_test_rmse, delimiter=",")
# %% ------------------------------------------------------------------------
# Test the best number of PCA
mse_array = []
for i in range(3, 30):
    pca = PCA(n_components=i,random_state=random_seed)
    Y_T_train_PCA = pca.fit_transform(Y_T_train)
    Y_T_test_PCA = pca.transform(Y_T_test)
    model.fit(X_train, Y_T_train_PCA)
    # should NOT use test to see best PC number
    Y_T_train_pred = model.predict(X_train)
    Y_T_train_pred = pca.inverse_transform(Y_T_train_pred)
    mse = mean_squared_error(Y_T_train, Y_T_train_pred)
    # Y_T_pred = model.predict(X_test)
    # Y_T_pred = pca.inverse_transform(Y_T_pred)
    # mse = mean_squared_error(Y_T_test, Y_T_pred)
    print(f'Training MSE =', mse)
    mse_array.append(mse)
# %% 
plt.plot(range(3, 30), mse_array)
plt.xlabel("Number of PCs")
plt.ylabel("MSE")
# * 27 PCs was the best

# %% ------------------------------------------------------------------------
# our model: one output is predicted by one base estimator
# model = MultiOutputRegressor(base_estimator)
model = XGBRegressor(tree_method='hist', multi_strategy='multi_output_tree')

# %% ------------------------------------------------------------------------
# train with the previous split
start_time = time.time()
model.fit(X_train, Y_T_train_PCA)
# model.fit(X_train, Y_T_train)
end_time = time.time()
print(end_time - start_time)
# 437.8546 (no PCA)
# 8.4737 (18 PCs)
# 12.2718 (27PCs)
# %% ------------------------------------------------------------------------
# test
Y_T_pred = model.predict(X_test)
Y_T_pred = pca.inverse_transform(Y_T_pred) # For pca'd results
# evaluate predictions
Y_T_train_pred = model.predict(X_train)
Y_T_train_pred = pca.inverse_transform(Y_T_train_pred) # For pca'd results
# %% ------------------------------------------------------------------------
mse = mean_squared_error(Y_T_train, Y_T_train_pred)
print(f'Training MSE =', mse)
mse = mean_squared_error(Y_T_test, Y_T_pred)
print(f'Test MSE =', mse)
# Training MSE = 0.0145520; Test MSE = 1.820630 (no PCA)
# Training MSE = 0.05830; Test MSE = 1.8999 (18 PCs)
# Training MSE = 0.02832; Test MSE = 1.84286 (27 PCs)
# After correct Ts, Ps, RHs
# Training MSE = 0.50818; Test MSE = 2.04469 (5 PCs)
# Training MSE = 0.04343; Test MSE = 2.10062 (18 PCs)
# %% ------------------------------------------------------------------------
eval_y_T_pred = model.predict(eval_x)
eval_y_T_pred = pca.inverse_transform(eval_y_T_pred) # For pca'd results
# %% ------------------------------------------------------------------------
# Save the result of XGBRegressor
np.savetxt("eval_y_Temp_XGBoost_pc18.csv", eval_y_T_pred, delimiter=",")
# %% ------------------------------------------------------------------------
# Create (prediction - truth) against altitude
Y_T_test_diff = Y_T_pred - Y_T_test
Y_T_test_sqrerr = Y_T_test_diff*Y_T_test_diff
Y_T_test_rmse = np.sqrt(np.mean(Y_T_test_sqrerr, axis=0)) # average over the row
# %% ------------------------------------------------------------------------
np.savetxt("rmse_T_XGB_pc18.csv", Y_T_test_rmse, delimiter=",")












# %% ------------------------------------------------------------------------
# grid search the hyperparameters
param_grid = {
                 'estimator__n_estimators': [5, 15, 25],
                 'estimator__max_depth': [10, 15, 20, 25],
                 'estimator__min_samples_split': [2, 5, 10],
                 'estimator__criterion': ['squared_error', 'absolute_error']
             }

# %% ------------------------------------------------------------------------
grid_rfr = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_rfr.fit(train_x, train_y_T)
# %% ------------------------------------------------------------------------
grid_rfr.best_params_
#
# %% ------------------------------------------------------------------------
# try the best estimator model
model = grid_rfr.best_estimator_
X_train, X_test, Y_T_train, Y_T_test = train_test_split(train_x, 
                                                        train_y_T,
                                                        test_size=0.2,
                                                        random_state=random_seed)
model.fit(X_train, Y_T_train)
# %% ------------------------------------------------------------------------
# evaluate predictions
accuracy = model.score(X_test, Y_T_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# %% ------------------------------------------------------------------------
