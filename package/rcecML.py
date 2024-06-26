# Author: Je-Ching Liao | Academia Sinica | University of Michigan
# Date: June 2024
# File Purpose: .py file package to apply machine learning to multi-input-multi- 
#               output tasks with Random Forest, XGBoost, or Neural Network
# %% ------------------------------------------------------------------------
import numpy as np
from enum import Enum
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from torch import nn, optim, from_numpy, cuda, tensor
import torch.nn.functional as F
import time
# %% ------------------------------------------------------------------------
class MIMORegressionMachineLearner:
    def __init__(self):
        self.random_state = None
        self.is_y_pca = False # Flag stating whether the y labels are pca'd
        self.isModelTrained = False # Flag indicating whether model is trained
        self.model_type = self.ModelType.Undeclared
        # Default grid for Random Forest
        self.rf_grid = {'n_estimators': 100,
                        'random_state' : self.random_state,
                        'criterion': 'squared_error',
                        'max_depth': None,
                        'min_sample_split': 2}
        # Default grid for XGBoost
        self.xgb_grid = {'random_state': self.random_state,
                         'tree_method': 'hist',
                         'multi_strategy': 'multi_output_tree'}

        # Note: attributes that are learned or are results from the machine
        # learning is suffixed by a "_"
        pass

    # Enum type of various model type
    class ModelType(Enum):
        Undeclared = 'undeclared'
        RandomForest = 'random_forest'
        XGBoost = 'xgboost'
        NeuralNetwork = 'neural_network'

    def randomStateSetter(self, random_state):
        # Check if random_state is indeed integer
        if not isinstance(random_state, int):
            raise Exception("Please specify an integer for random state")
        self.random_state = random_state

    def dataLoader(self, var_x, var_y, test_size=None):
        # Check if random state is set
        if self.random_state == None:
            raise Exception("Please specify a random state by using \
                             randomStateSetter(); an integer is preferred")
        
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
            raise Exception("Please specify a random state by using \
                             randomStateSetter(); an integer is preferred")
        
        # Check if number of PCs is specified
        if n_components == None:
            raise Exception("Please specify the number of principal components")
        
        self.pca_ = PCA(n_components=n_components,
                        random_state=self.random_state)
        self.train_y = self.pca_.fit_transform(self.train_y)
        self.test_y = self.pca_.transform(self.test_y)
        self.is_y_pca = True
    
    def hyperparamSetter(self, param_name, param_val):
        if self.model_type == self.ModelType.RandomForest:
            self.rf_grid[param_name] = param_val # Add or modify a param
            self.model.set_params(**self.rf_grid) # Set the parameters

        elif self.model_type == self.ModelType.XGBoost:
            self.xgb_grid[param_name] = param_val # Add or modify a param
            self.model.set_params(**self.rf_grid) # Set the parameters

        elif self.model_type == self.ModelType.Undeclared:
            raise Exception("Please initialize a model first before setting \
                            the hyperparameters")

    # Set up the random forest regressor for multiple output
    def randomForestRegressorCustom(self):
        # Check if random state is set
        if self.random_state == None:
            raise Exception("Please specify a random state by using \
                             randomStateSetter(); an integer is preferred")
        
        self.model = MultiOutputRegressor(RandomForestRegressor())
        self.model_type = self.ModelType.RandomForest # Set the flag
        self.model.set_params(**self.rf_grid) # Set the parameters

    # Set up the XGBoosting regressor for multiple output
    def XGBRegressorCustom(self):
        # Check if random state is set
        if self.random_state == None:
            raise Exception("Please specify a random state by using \
                             randomStateSetter(); an integer is preferred")
        
        self.model = XGBRegressor()
        self.model_type = self.ModelType.XGBoost # Set the flag
        self.model.set_params(**self.xgb_grid) # Set the parameters
        
    def trainTheModel(self):
        # Check if a model has been chosen
        if not hasattr(self, 'model'):
            raise Exception("Please specify and tune the model before training")
        
        start_time = time.time()
        # Train the model with the x and y of training data
        self.model.fit(self.train_x, self.train_y)
        self.isModelTrained = True # Flag indicating the model has been trained
        end_time = time.time()

        self.training_time = end_time - start_time # record the training time

    def predict(self):
        # Check if the model has been training
        if not self.isModelTrained:
            raise Exception("Please train the model with trainTheModel() first")
        
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
    
    def predictWithNewData(self, new_data_x):
        # Check if the model has been training
        if not self.isModelTrained:
            raise Exception("Please train the model with trainTheModel() first")
        
        # Check if new data has same number of columns as training labels
        if new_data_x.shape[1] != self.train_x[1]:
            raise Exception(f"Please make sure the number of columns \
                            ({new_data_x.shape[1]}) of the new data is same \
                            as that ({self.train_x[1]}) of the training x data")
        
        # if the y is pca'd, this block inverse transform them back
        if self.is_y_pca:
            self.pred_new_data_output_ = self.pca_.inverse_transform(
                self.model.predict(new_data_x))
        else:
            self.pred_new_data_output_ = self.model.predict(new_data_x)

    def rmseOverTrain(self, isPercentage=True):
        if not hasattr(self, 'pred_train_output_'):
            raise Exception("Please run predict() function first, to create \
                            prediction data from the x features")
        
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
            raise Exception("Please run predict() function first, to create \
                            prediction data from the x features")
        
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
            raise Exception("Please run predictWithNewData() function first, \
                            to predict from the new data")
        if new_data_y == None:
            raise Exception("RMSE can only be calculated between prediction \
                            and given ground truth labels; please provide a \
                            ground truth label data into the function")
        
        # isPercentage being True means that the RMSE will be calculated as 
        # (pred - truth)/truth, while isPercentage being False means that the 
        # RMSE is calculated as (pred - truth)

        diff = self.pred_new_data_output_ - new_data_y
        if isPercentage:
            diff = diff / self.new_data_y 
        squerr = diff * diff
        return np.sqrt(np.mean(squerr, axis=0)) # average over the rows


    # Getter functions below (Used to get the values)
    def train_mse(self):
        return self.train_mse_loss_
    
    def test_mse(self):
        return self.test_mse_loss_
    
    def prediction_from_train(self):
        return self.pred_train_output_
    
    def prediction_from_test(self):
        return self.pred_test_output_
    
    def prediction_from_new_data(self):
        return self.pred_new_data_output_
