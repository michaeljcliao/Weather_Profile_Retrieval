# Author: Je-Ching Liao | Academia Sinica | University of Michigan
# Date: June 2024
# File Purpose: .py file package to apply machine learning to multi-input-multi- 
#               output tasks with Random Forest, XGBoost, or Neural Network
# %% ------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from torch import nn, optim, from_numpy, cuda
import time
# %% ------------------------------------------------------------------------
class MIMORegressionMachineLearner:
    def __init__(self):
        self.random_state = None
        self.is_y_pca = False # Flag stating whether the y labels are pca'd
        # Number of principal components: set as 1 to initialize
        self.n_components = 1 
        self.isModelTrained = False # Flag indicating whether model is trained
        self.model_type = self.ModelType.Undeclared
        # Default grid for Random Forest
        self.rf_grid = {'n_estimators': 100,
                        'random_state' : self.random_state,
                        'criterion': 'squared_error',
                        'max_depth': None,
                        'min_samples_split': 2}
        # Default grid for XGBoost
        self.xgb_grid = {'random_state': self.random_state,
                         'tree_method': 'hist',
                         'multi_strategy': 'multi_output_tree'}
        
        self.neuralNet = self.Net(self)
        self.nn_layer_array = ['l1', 'l2', 'outlayer']

        # Note: attributes that are learned or are results from the machine
        # learning is suffixed by a "_"
        pass

    # Enum type of various model type
    class ModelType(Enum):
        Undeclared = 'undeclared'
        RandomForest = 'random_forest'
        XGBoost = 'xgboost'
        NeuralNetwork = 'neural_network'

    class Net(nn.Module):

        def __init__(self, parent):
            super(parent.Net, self).__init__()
            self.parent = parent
            self.l1 = nn.Sequential(
                nn.Linear(37, 30),
                nn.ReLU(inplace=True)
            )
            self.l2 = nn.Sequential(
                nn.Linear(30, 20),
                nn.ReLU(inplace=True)
            )
            self.outlayer = nn.Linear(20, parent.n_components)
            self.double()

        def forward(self, x):
            x = self.l1(x)
            x = self.l2(x)
            return self.outlayer(x)

    def randomStateSetter(self, random_state):
        # Check if random_state is indeed integer
        if not isinstance(random_state, int):
            raise Exception("Please specify an integer for random state")
        self.random_state = random_state

    def dataLoader(self, var_x, var_y, test_size=None):
        # Check if random state is set
        if self.random_state == None:
            raise Exception("Please specify a random state by using"
                            " randomStateSetter(); an integer is preferred")
        
        # Check if split ratio is specified
        if test_size == None:
            raise Exception("Please specify a test size between 0 and 1")
        
        # Load data
        data_x = np.loadtxt(var_x, delimiter=',', dtype=np.float64)
        data_y = np.loadtxt(var_y, delimiter=',', dtype=np.float64)

        # Split into train and test
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            data_x, data_y, test_size=test_size, random_state=self.random_state)
        
    def standardizeX(self):
        std_scaler = StandardScaler()
        self.train_x = std_scaler.fit_transform(self.train_x)
        self.test_x = std_scaler.transform(self.test_x)

    def pcaY(self, n_components=None):
        # Check if random state is set
        if self.random_state == None:
            raise Exception("Please specify a random state by using"
                            " randomStateSetter(); an integer is preferred")
        
        # Check if number of PCs is specified
        if n_components == None:
            raise Exception("Please specify the number of principal components")
        
        self.n_components = n_components        
        self.pca_ = PCA(n_components=n_components,
                        random_state=self.random_state)
        self.train_y_pca_ = self.pca_.fit_transform(self.train_y)
        self.test_y_pca_ = self.pca_.transform(self.test_y)
        self.is_y_pca = True
    
    def hyperparamSetter(self, param_name, param_val):
        if self.model_type == self.ModelType.RandomForest:
            model = RandomForestRegressor()
            model.set_params(**self.rf_grid) # Set the parameters
            self.model = MultiOutputRegressor(model)

        elif self.model_type == self.ModelType.XGBoost:
            self.xgb_grid[param_name] = param_val # Add or modify a param
            self.model.set_params(**self.rf_grid) # Set the parameters

        elif self.model_type == self.ModelType.Undeclared:
            raise Exception("Please initialize a model first before setting"
                            " the hyperparameters")

    # Set up the random forest regressor for multiple output
    def randomForestRegressorInitialize(self):
        # Check if random state is set
        if self.random_state == None:
            raise Exception("Please specify a random state by using"
                            " randomStateSetter(); an integer is preferred")
        
        self.model_type = self.ModelType.RandomForest # Set the flag
        model = RandomForestRegressor()
        model.set_params(**self.rf_grid) # Set the parameters
        self.model = MultiOutputRegressor(model)

    # Set up the XGBoosting regressor for multiple output
    def XGBRegressorInitialize(self):
        # Check if random state is set
        if self.random_state == None:
            raise Exception("Please specify a random state by using"
                            " randomStateSetter(); an integer is preferred")
        
        self.model_type = self.ModelType.XGBoost # Set the flag
        self.model = XGBRegressor()
        self.model.set_params(**self.xgb_grid) # Set the parameters

    def neuralNetworkInitialize(self, clip_value=5, isClipped=True):
        self.model_type = self.ModelType.NeuralNetwork # Set the flag

        # If the dataset does not undergo PCA, the n_components in Net() is 
        # the original dimension of the y label data
        if not self.is_y_pca:
            self.n_components = self.train_y.shape[1]
        
        self.model = self.Net(self)
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(device)
        
        # The value for the max_norm in torch.nn.utils.clip_grad_norm_(), the 
        # smaller, the more radical it is going to reduce the gradient
        self.clip_value = clip_value
        self.isClipped = isClipped

    def neuralNetworkAddLayer(self):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")

    def neuralNetworkRemoveLayer(self):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")

    def neuralNetworkModifyNeuron(self, numLayer=None):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")
    
    def neuralNetworkModifyActivation(self,
                                      numLayer=None,
                                      activation_function=None):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")
        
        if numLayer == None:
            raise Exception("Please specify the layer to modify with an integer")
        
        if activation_function == None:
            raise Exception("Please specify the desired activation function")
        
        layer_name = self.nn_layer_array[numLayer-1]
        self.neuralNet.layer_name
        # TODO: Check ChatGPT for its recommedation
        # !

    def neuralNetworkCriterionSetter(self, criterion=None):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")
        
        if criterion == None:
            criterion = nn.MSELoss(reduction='sum')

        self.criterion = criterion

    def neuralNetworkOptimizerSetter(self, optimizer=None):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")
        
        if optimizer == None:
            optimizer = optim.AdamW(self.model.parameters(), lr=0.01)

        self.optimizer = optimizer

        
    def trainRFXGB(self):
        # Check if a model has been chosen
        if not hasattr(self, 'model'):
            raise Exception("Please specify and tune the model before training")
        
        if self.model_type == self.ModelType.NeuralNetwork:
            raise Exception("Please use trainNN() function")
        
        start_time = time.time()
        # Train the model with the x and y of training data
        # Two cases: pca'd or not
        if self.is_y_pca:
            self.model.fit(self.train_x, self.train_y_pca_)
        else:
            self.model.fit(self.train_x, self.train_y)
        self.isModelTrained = True # Flag indicating the model has been trained
        end_time = time.time()

        self.training_time_ = end_time - start_time # record the training time

    def trainNN(self, epochs=None):
        # Check if a model has been chosen
        if not hasattr(self, 'model'):
            raise Exception("Please specify and tune the model before training")
        
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Please use trainRFXGB() function")
        
        if epochs == None:
            raise Exception("Please specify a number of epochs, the number of"
                            " times this Neural Network will be trained")

        # Training loop
        loss_array = []
        train_mse_arr = []
        test_mse_arr = []

        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        start_time = time.time()
        for epoch in range(epochs):
            # 1) Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(from_numpy(self.train_x))

            # 2) Compute and print loss
            loss = self.criterion(y_pred, from_numpy(self.train_y_pca_))
            print(f'Epoch: {epoch} | Loss: {loss.item()} ')
            loss_array.append(loss.item())

            if epoch % 100 == 99:
                Y_train_pred = self.model(from_numpy(self.train_x))
                Y_train_pred = Y_train_pred.detach().numpy()

                if self.is_y_pca:
                    Y_train_pred = self.pca_.inverse_transform(Y_train_pred)

                train_mse = mean_squared_error(self.train_y, Y_train_pred)
                train_mse_arr.append(train_mse)

                Y_test_pred = self.model(from_numpy(self.test_x))
                Y_test_pred = Y_test_pred.detach().numpy()

                if self.is_y_pca:
                    Y_test_pred = self.pca_.inverse_transform(Y_test_pred)
                
                test_mse = mean_squared_error(self.test_y, Y_test_pred)
                test_mse_arr.append(test_mse)

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            # Clip the gradient if it is enabled
            if self.isClipped:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.clip_value)
            self.optimizer.step()
        end_time = time.time()
        
        self.isModelTrained = True # Flag indicating the model has been trained
        self.training_time_ = end_time - start_time # record the training time
        self.loss_array_ = loss_array
        self.train_mse_array_ = train_mse_arr
        self.test_mse_array_ = test_mse_arr

    def predictRFXGB(self):
        # Check if the model has been training
        if not self.isModelTrained:
            raise Exception("Please train the model with trainRFXGB() first")
        
        if self.model_type == self.ModelType.NeuralNetwork:
            raise Exception("Please use predictNN() function")
        
        # Use the trained model to predict labels with data fed into the 
        # self.model.predict() function
        # if the y is pca'd, this block inverse transform them back
        if self.is_y_pca:
            self.pred_train_output_ = self.pca_.inverse_transform(
                self.model.predict(self.train_x))
            self.pred_test_output_ = self.pca_.inverse_transform(
                self.model.predict(self.test_x))
        else:
            self.pred_train_output_ = self.model.predict(self.train_x)
            self.pred_test_output_ = self.model.predict(self.test_x)

        # Calculate the mean squared error with both prediction from training
        # and test dataset
        self.train_mse_loss_ = mean_squared_error(self.train_y, 
                                                  self.pred_train_output_)
        self.test_mse_loss_ = mean_squared_error(self.test_y, 
                                                 self.pred_test_output_)
        
    def predictNN(self):
        # Check if the model has been training
        if not self.isModelTrained:
            raise Exception("Please train the model with trainNN() first")
        
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Please use predictRFXGB() function")
        
        # Use the trained model to predict labels with data fed into the 
        # self.model() function
        # if the y is pca'd, this block inverse transform them back
        if self.is_y_pca:
            self.pred_train_output_ = self.pca_.inverse_transform(
                self.model(from_numpy(self.train_x)).detach().numpy())
            self.pred_test_output_ = self.pca_.inverse_transform(
                self.model(from_numpy(self.test_x)).detach().numpy())
        else:
            self.pred_train_output_ = self.model(
                from_numpy(self.train_x)).detach().numpy()
            self.pred_test_output_ = self.model(
                from_numpy(self.test_x)).detach().numpy()
        

        # Calculate the mean squared error with both prediction from training
        # and test dataset
        self.train_mse_loss_ = mean_squared_error(self.train_y, 
                                                  self.pred_train_output_)
        self.test_mse_loss_ = mean_squared_error(self.test_y, 
                                                 self.pred_test_output_)
    
    def predictNewDataRFXGB(self, new_data_x):
        # Check if the model has been training
        if not self.isModelTrained:
            raise Exception("Please train the model with trainTheModel() first")
        
        if self.model_type == self.ModelType.NeuralNetwork:
            raise Exception("Please use predictNewDataNN() function")
        
        # Check if new data has same number of columns as training labels
        if new_data_x.shape[1] != self.train_x[1]:
            raise Exception("Please make sure the number of columns of the new"
                            " data is same as that of the training x data")
        
        # if the y is pca'd, this block inverse transform them back
        if self.is_y_pca:
            self.pred_new_data_output_ = self.pca_.inverse_transform(
                self.model.predict(new_data_x))
        else:
            self.pred_new_data_output_ = self.model.predict(new_data_x)

    def predictNewDataNN(self, new_data_x):
        # Check if the model has been training
        if not self.isModelTrained:
            raise Exception("Please train the model with trainTheModel() first")
        
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Please use predictNewDataRFXGB() function")
        
        # Check if new data has same number of columns as training labels
        if new_data_x.shape[1] != self.train_x[1]:
            raise Exception("Please make sure the number of columns of the new"
                            " data is same as that of the training x data")
        
        # if the y is pca'd, this block inverse transform them back
        if self.is_y_pca:
            self.pred_new_data_output_ = self.pca_.inverse_transform(
                self.model(from_numpy(new_data_x)).detach().numpy())
        else:
            self.pred_new_data_output_ = self.model(
                from_numpy(new_data_x)).detach().numpy()

    def rmseOverTrain(self, isPercentage=True):
        if not hasattr(self, 'pred_train_output_'):
            raise Exception("Please run predict() function first, to create"
                            " prediction data from the x features")
        
        # isPercentage being True means that the RMSE will be calculated as 
        # (pred - truth)/truth, while isPercentage being False means that the 
        # RMSE is calculated as (pred - truth)

        diff = self.pred_train_output_ - self.train_y
        if isPercentage:
            diff = diff / self.train_y 
        squerr = diff * diff
        return np.sqrt(np.mean(squerr, axis=0)) # average over the rows
    
    def rmseOverTest(self, isPercentage=True):
        if not hasattr(self, 'pred_test_output_'):
            raise Exception("Please run predict() function first, to create"
                            " prediction data from the x features")
        
        # isPercentage being True means that the RMSE will be calculated as 
        # (pred - truth)/truth, while isPercentage being False means that the 
        # RMSE is calculated as (pred - truth)

        diff = self.pred_test_output_ - self.test_y
        if isPercentage:
            diff = diff / self.test_y 
        squerr = diff * diff
        return np.sqrt(np.mean(squerr, axis=0)) # average over the rows
    
    def rmseOverNewData(self, new_data_y=None, isPercentage=True):
        if not hasattr(self, 'pred_new_data_output_'):
            raise Exception("Please run predictWithNewData() function first"
                            " to predict from the new data")
        if new_data_y == None:
            raise Exception("RMSE can only be calculated between prediction"
                            " and given ground truth labels; please provide a"
                            " ground truth label data into the function")
        
        # isPercentage being True means that the RMSE will be calculated as 
        # (pred - truth)/truth, while isPercentage being False means that the 
        # RMSE is calculated as (pred - truth)

        diff = self.pred_new_data_output_ - new_data_y
        if isPercentage:
            diff = diff / self.new_data_y 
        squerr = diff * diff
        return np.sqrt(np.mean(squerr, axis=0)) # average over the rows
        
    def plotLossNN(self,
                   xlabel='Epoch #',
                   ylabel='Neural Network Loss',
                   title='Neural Network Loss Over Training Epochs'):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("This plot only works with Neural Network")
        
        # Check if the model has been training
        if not self.isModelTrained:
            raise Exception("Please train the model with trainNN() first")
        
        plt.plot(self.loss_array_)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plotMSENN(self,
                  xlabel='Epoch # (*100)',
                  ylabel='Mean-Squared Error',
                  title='Mean-Squared Error Over Training Epochs'):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("This plot only works with Neural Network")
        
        # Check if the model has been training
        if not self.isModelTrained:
            raise Exception("Please train the model with trainNN() first")
        
        plt.plot(self.train_mse_array_, label='train MSE')
        plt.plot(self.test_mse_array_, label='test MSE')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()