# %% ------------------------------------------------------------------------
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch import nn, optim, from_numpy, cuda, tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
# %% ------------------------------------------------------------------------
os.chdir('/NFS15/michaelliao/retrievaldata')
# %% ------------------------------------------------------------------------
random_seed = 7036
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training model on {device}\n{"=" * 44}')
# %% ------------------------------------------------------------------------
train_x = np.loadtxt('./train_x.csv', delimiter=',', dtype=np.float64)
train_y_T = np.loadtxt('./train_y_Temp.csv', delimiter=',', dtype=np.float64)
# train_x = train_x[:, :-1]
print(train_x.shape)
print(train_y_T.shape)

# %% ------------------------------------------------------------------------
# data split
X_train, X_test, Y_T_train, Y_T_test = train_test_split(train_x, 
                                                        train_y_T,
                                                        test_size=0.2,
                                                        random_state=random_seed)

print(Y_T_train.shape)
# %% ------------------------------------------------------------------------
# standardize the input
stdScaler = StandardScaler()
X_train = from_numpy(stdScaler.fit_transform(X_train))
X_test = from_numpy(stdScaler.transform(X_test))
# %% ------------------------------------------------------------------------
# PCA reduce dimensionality
numPC = 27
pca = PCA(n_components=numPC, random_state=random_seed)

Y_T_train_PCA = pca.fit_transform(Y_T_train)
Y_T_test_PCA = pca.transform(Y_T_test)
Y_T_train_PCA = from_numpy(Y_T_train_PCA)
Y_T_test_PCA = from_numpy(Y_T_test_PCA)
print(Y_T_train_PCA.shape)
# %% ------------------------------------------------------------------------
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(37, 30)
        self.l2 = nn.Linear(30, 20)
        self.l3 = nn.Linear(20, numPC)
        self.double()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
# %% ------------------------------------------------------------------------
# %% ------------------------------------------------------------------------
# our model
model = Net()
model.to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# %% ------------------------------------------------------------------------
# Training loop
clip_value = 5
loss_array = []
train_mse_arr = []
test_mse_arr = []
best_loss = 10**7
loss_stagnate_count = 0

torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
start_time = time.time()
for epoch in range(7000):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred_PCA = model(X_train)

    # 2) Compute and print loss
    loss = criterion(y_pred_PCA, Y_T_train_PCA)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')
    loss_array.append(loss.item())

    # if loss.item() < best_loss:
    #     best_loss = loss.item()
    # elif loss_stagnate_count < 150:
    #     loss_stagnate_count += 1
    # else:
    #     break

    if epoch % 100 == 99:
        Y_T_train_pred_PCA = model(X_train)
        Y_T_train_pred_PCA = Y_T_train_pred_PCA.detach().numpy()
        Y_T_train_pred = pca.inverse_transform(Y_T_train_pred_PCA)
        train_mse = mean_squared_error(Y_T_train, Y_T_train_pred)
        train_mse_arr.append(train_mse)

        Y_T_test_pred_PCA = model(X_test)
        Y_T_test_pred_PCA = Y_T_test_pred_PCA.detach().numpy()
        Y_T_test_pred = pca.inverse_transform(Y_T_test_pred_PCA)
        test_mse = mean_squared_error(Y_T_test, Y_T_test_pred)
        test_mse_arr.append(test_mse)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    # gradient clipping!
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
end_time = time.time()
print(end_time - start_time)
# %% ------------------------------------------------------------------------
# %% ------------------------------------------------------------------------
# Draw Loss Graph
plt.plot(loss_array)
# plt.xlim([1000, 7000])
# plt.ylim([0,30000])
plt.xlabel('Epoch #')
plt.title('Neural Network Loss')
plt.show()
# %% ------------------------------------------------------------------------
# range(100,2600,100), 
plt.plot(train_mse_arr, label='train MSE')
plt.plot(test_mse_arr, label='test MSE')
# plt.xlim([1000, 5000])
plt.xlabel('Epoch # (*100)')
plt.title(f'Mean-Squared Error with %d PC Predictands' % numPC)
plt.legend()
plt.show()
# %% ------------------------------------------------------------------------
# Check the training set MSE
Y_T_train_pred_PCA = model(X_train)
Y_T_train_pred_PCA = Y_T_train_pred_PCA.detach().numpy()
Y_T_train_pred = pca.inverse_transform(Y_T_train_pred_PCA) # For pca'd results
# %% ------------------------------------------------------------------------
mse = mean_squared_error(Y_T_train, Y_T_train_pred)
print(f'Training MSE =', mse)
# After new Ts, Ps, RHs
# 0.51572 (5 PCs)
# 0.19997 (10 PCs)
# 0.15715 (18 PCs)
# 0.13711 (27 PCs)
# %% ------------------------------------------------------------------------
# test
Y_T_test_pred_PCA = model(X_test)
Y_T_test_pred_PCA = Y_T_test_pred_PCA.detach().numpy()
Y_T_test_pred = pca.inverse_transform(Y_T_test_pred_PCA) # For pca'd results
# %% ------------------------------------------------------------------------
# Store selected prediction from train and test datasets
np.savetxt("train_pred_T_NN_pc10.csv", Y_T_train_pred, delimiter=",")
np.savetxt("test_pred_T_NN_pc10.csv", Y_T_test_pred, delimiter=",")
# %% ------------------------------------------------------------------------

# evaluate predictions
mse = mean_squared_error(Y_T_test, Y_T_test_pred)
print(f'Test MSE =', mse)
# After new Ts, Ps, RHs
# 1.72918 (5 PCs)
# 1.59991 (10 PCs)
# 1.71983 (18 PCs)
# 1.67712 (27 PCs)
# %% ------------------------------------------------------------------------
# Create (prediction - truth) against altitude
Y_T_test_diff = Y_T_test_pred - Y_T_test
Y_T_test_sqrerr = Y_T_test_diff*Y_T_test_diff
Y_T_test_rmse = np.sqrt(np.mean(Y_T_test_sqrerr, axis=0)) # average over the row
# %% ------------------------------------------------------------------------
# np.savetxt("rmse_T_NN_pc27.csv", Y_T_test_rmse, delimiter=",")
# %% ------------------------------------------------------------------------
train_y_T_lv2 = np.loadtxt('./train_y_Temp_lv2.csv',
                           delimiter=',',
                           dtype=np.float64)
X_train, X_test, Y_T_train_lv2, Y_T_test_lv2 = train_test_split(train_x, 
                                                        train_y_T_lv2,
                                                        test_size=0.2,
                                                        random_state=random_seed)
# %% ------------------------------------------------------------------------
mse = mean_squared_error(Y_T_train[:, 0:1000], Y_T_train_lv2[:, 0:1000])
print(f'Training MSE =', mse)
mse = mean_squared_error(Y_T_test[:, 0:1000], Y_T_test_lv2[:, 0:1000])
print(f'Test MSE =', mse)
# %% ------------------------------------------------------------------------
# Create (prediction - truth) against altitude for lvl2
Y_T_test_lv2_diff = Y_T_test_lv2 - Y_T_test
Y_T_test_lv2_sqrerr = Y_T_test_lv2_diff*Y_T_test_lv2_diff
Y_T_test_lv2_rmse = np.sqrt(np.mean(Y_T_test_lv2_sqrerr, axis=0)) # average over the row
np.savetxt("rmse_T_lv2.csv", Y_T_test_lv2_rmse, delimiter=",")
# %% ------------------------------------------------------------------------
train_y_T_original = np.loadtxt('./train_y_Temp_original.csv',
                               delimiter=',',
                               dtype=np.float64)
train_y_T_lv2_original = np.loadtxt('./train_y_Temp_lv2_original.csv',
                                   delimiter=',',
                                   dtype=np.float64)
# %% ------------------------------------------------------------------------
# find an array in a 2d array
np.argwhere(np.isin(Y_T_train[:, 0:1000], train_y_T_original[3, 0:1000]).all(axis=1))
# %% ------------------------------------------------------------------------
plt.plot(train_y_T_original[2, 0:1000], label = 'ground truth')
plt.plot(Y_T_train_pred[18, 0:1000], label = 'my pred')
plt.plot(train_y_T_lv2_original[2, 0:1000], label = 'old pred')
plt.legend()
plt.show()
mse = mean_squared_error(train_y_T_original[3, 0:1000], train_y_T_lv2_original[3, 0:1000])
print(f'Old ML MSE =', mse)
mse = mean_squared_error(train_y_T_original[3, 0:1000], Y_T_train_pred[18, 0:1000])
print(f'my MSE =', mse)
# %% ------------------------------------------------------------------------
# final column for lv2 is nan (1000-th)
# ! THIS MSE IS EXTREMELY HIGH
mse = mean_squared_error(train_y_T_original[:, 0:1000], train_y_T_lv2_original[:, 0:1000])
print(f'Old ML MSE =', mse)
# %% ------------------------------------------------------------------------
# Use eval data set
eval_x = np.loadtxt('./eval_x.csv', delimiter=',', dtype=np.float64)
eval_x = from_numpy(stdScaler.transform(eval_x))
# %% ------------------------------------------------------------------------
# Predict with eval data set
eval_y_T_pred_PCA = model(eval_x)
eval_y_T_pred_PCA = eval_y_T_pred_PCA.detach().numpy()
eval_y_T_pred = pca.inverse_transform(eval_y_T_pred_PCA)
# %% ------------------------------------------------------------------------
# Save the eval result of Neural Network
np.savetxt("eval_y_Temp_NN_pc10.csv", eval_y_T_pred, delimiter=",")
# %%
