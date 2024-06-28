# Author: Je-Ching Liao | Academia Sinica | University of Michigan
# Date: June 2024
# File Purpose: .py file package to apply machine learning to multi-input-multi- 
#               output tasks with Random Forest, XGBoost, or Neural Network

import random
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
from torch import nn, optim, as_tensor, cuda
import time

class MIMORegressionMachineLearner:
    def __init__(self):
        self.random_state = 7036
        self.is_y_pca = False # Flag stating whether the y labels are pca'd
        
        # Dimension of the input: set as 1 to initialize
        self.x_input_dim = 1

        # Final output dimension: set as 1 to initialize; could be affected by
        # applying pca to y
        self.y_output_dim = 1
        
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
        # Default arrangement of Neural Network
        self.nn_neuron_list = [self.x_input_dim, 30, 20, self.y_output_dim]
        self.nn_activation_list = [nn.ReLU(), nn.ReLU()]
        self.neuralNet = self.Net(self,
                                  self.nn_neuron_list,
                                  self.nn_activation_list)

        # Set the random seeds
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

        def __init__(self, parent, neuron_list, activation_list):
            super(parent.Net, self).__init__()
            self.parent = parent
            self.neuron_list = neuron_list
            self.activation_list = activation_list
            self.layers = nn.ModuleList()
            for i in range(len(neuron_list)-2):
                self.layers.append(nn.Sequential(
                    nn.Linear(self.neuron_list[i],
                              self.neuron_list[i+1]),
                    self.activation_list[i]
                ))
            # Add the output layer (without activation)
            self.layers.append(nn.Linear(self.neuron_list[-2],
                                         self.neuron_list[-1]))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
        
    def refreshNNNeuronList(self):
        # Updates the x input dim and y output dim into the list and into the 
        # Neural Network model when the model is initialized
        self.nn_neuron_list = [self.x_input_dim, 30, 20, self.y_output_dim]

    def dataLoader(self, var_x, var_y, test_size=None):      
        # Check if split ratio is specified
        if test_size == None:
            raise Exception("Please specify a test size between 0 and 1")
        
        # Load data
        data_x = np.loadtxt(var_x, delimiter=',', dtype=np.float32)
        data_y = np.loadtxt(var_y, delimiter=',', dtype=np.float32)

        self.x_input_dim = data_x.shape[1] # Store the input x dimension
        self.refreshNNNeuronList()

        # Split into train and test
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            data_x, data_y, test_size=test_size, random_state=self.random_state)
        
    def standardizeX(self):
        std_scaler = StandardScaler()
        self.train_x = std_scaler.fit_transform(self.train_x)
        self.test_x = std_scaler.transform(self.test_x)

    def pcaY(self, n_components=None):      
        # Check if number of PCs is specified
        if n_components == None:
            raise Exception("Please specify the number of principal components")
        
        self.y_output_dim = n_components
        self.refreshNNNeuronList()  
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
        self.model_type = self.ModelType.RandomForest # Set the flag
        model = RandomForestRegressor()
        model.set_params(**self.rf_grid) # Set the parameters
        self.model = MultiOutputRegressor(model)

    # Set up the XGBoosting regressor for multiple output
    def XGBRegressorInitialize(self):      
        self.model_type = self.ModelType.XGBoost # Set the flag
        self.model = XGBRegressor()
        self.model.set_params(**self.xgb_grid) # Set the parameters

    def neuralNetworkInitialize(self,
                                clip_value=5,
                                isClipped=True,
                                neuron_list=None,
                                activation_list=None,
                                criterion=None,
                                optimizer=None):
        self.model_type = self.ModelType.NeuralNetwork # Set the flag

        # If the dataset does not undergo PCA, the n_components in Net() is 
        # the original dimension of the y label data
        if not self.is_y_pca:
            self.y_output_dim = self.train_y.shape[1]
            self.refreshNNNeuronList()

        # Set the output dimension of the Net as n_components 
        # self.neuralNet.outlayer.out_features = self.y_output_dim

        # If layer and activation are designated, change it
        if (neuron_list != None and activation_list != None):
            self.nn_neuron_list = neuron_list
            self.nn_activation_list = activation_list
        elif (neuron_list != None or activation_list != None):
            raise Exception("Please set customized neuron_list and activation"
                            " list simultaneously to ensure a correct setup")

        self.model = self.Net(self, self.nn_neuron_list, self.nn_activation_list)
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(device)
        
        # The value for the max_norm in torch.nn.utils.clip_grad_norm_(), the 
        # smaller, the more radical it is going to reduce the gradient
        self.clip_value = clip_value
        self.isClipped = isClipped

        self.neuralNetworkCriterionSetter(criterion)
        self.neuralNetworkOptimizerSetter(optimizer)

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

    def neuralNetworkAddLayer(self,
                              prev_layer=None,
                              in_neurons=None,
                              out_neurons=None,
                              activation_function=nn.ReLU()):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")
        
        if prev_layer == None:
            raise Exception("Please specify the previous layer, i.e., after"
                            " which the new layer will be added")  
        elif prev_layer < 1:
            raise Exception("Invalid previous layer: must be integer larger"
                            " than or equal to 1")
        elif prev_layer > len(self.nn_activation_list):
            raise Exception("Invalid layer number: too large")
        
        if ((in_neurons == None and out_neurons == None) or 
            (in_neurons != None and out_neurons != None)):
            raise Exception("Please specify one and only one change with an"
                            " integer: dimension of input or dimension of output")
        
        # Set the number of neurons for the new layer
        if in_neurons == None:
            self.nn_neuron_list.insert(prev_layer+1, out_neurons)

        if out_neurons == None:
            self.nn_neuron_list.insert(prev_layer, in_neurons)
        # Set the activation function for the new layer
        self.nn_activation_list.insert(prev_layer, activation_function)

        # Re-initialize
        self.neuralNetworkInitialize()
        

    def neuralNetworkRemoveLayer(self, num_layer=None, isPrevPreserved=True):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")
        if num_layer == None:
            raise Exception("Please specify the layer to remove with an integer")
        elif num_layer < 1:
            raise Exception("Invalid layer number: must be integer larger than"
                            " or equal to 1")
        elif num_layer > len(self.nn_activation_list):
            raise Exception("Invalid layer number: too large")

        if isPrevPreserved:
            # isPrevPreserved being True: the output dim of the previous layer
            # is preserved, while changing the input dim of the next layer
            self.nn_neuron_list.pop(num_layer)
        else:
            # isPrevPreserved being False: the input dim of the next layer is
            # preserved, while changing the output dim of the previous layer
            self.nn_neuron_list.pop(num_layer-1)
        # Remove the activation function for the layer
        self.nn_activation_list.pop(num_layer-1)
        # Re-initialize
        self.neuralNetworkInitialize()
        

    def neuralNetworkModifyNeuron(self,
                                  num_layer=None,
                                  in_neurons=None,
                                  out_neurons=None):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")

        if num_layer == None:
            raise Exception("Please specify the layer to modify with an integer")
        elif num_layer < 1:
            raise Exception("Invalid layer number: must be integer larger than"
                            " or equal to 1")
        elif num_layer > len(self.nn_activation_list):
            raise Exception("Invalid layer number: too large")
        
        if (in_neurons == None and out_neurons == None):
            raise Exception("Please specify at least one change with integer:"
                            " dimension of input or dimension of output")
        
        # Change the number of neurons
        if in_neurons != None:
            self.nn_neuron_list[num_layer-1] = in_neurons
        if out_neurons != None:
            self.nn_neuron_list[num_layer] = out_neurons

        # Re-initialize
        self.neuralNetworkInitialize()
    
    def neuralNetworkModifyActivation(self,
                                      num_layer=None,
                                      activation_function=None):
        if self.model_type != self.ModelType.NeuralNetwork:
            raise Exception("Neural Network Exclusive Function")
        
        if num_layer == None:
            raise Exception("Please specify the layer to modify with an integer")
        elif num_layer < 1:
            raise Exception("Invalid layer number: must be integer larger than"
                            " or equal to 1")
        elif num_layer > len(self.nn_activation_list):
            raise Exception("Invalid layer number: too large")
        
        if activation_function == None:
            raise Exception("Please specify the desired activation function")
        
        self.nn_activation_list[num_layer-1] = activation_function
        # Re-initialize
        self.neuralNetworkInitialize()

    def neuralNetworkVisualizer(self):
        for module in self.model.modules():
            print(module)
            break # Effectively only printing the Net()
        
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

        train_x = as_tensor(self.train_x, dtype=torch.float32)
        train_y = as_tensor(
            self.train_y_pca_ if self.is_y_pca else self.train_y,
            dtype=torch.float32)
        test_x = as_tensor(self.test_x, dtype=torch.float32)

        start_time = time.time()
        for epoch in range(epochs):
            # 1) Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(train_x)

            # 2) Compute and print loss
            loss = self.criterion(y_pred, train_y)
            print(f'Epoch: {epoch} | Loss: {loss.item()} ')
            loss_array.append(loss.item())

            if epoch % 100 == 99:
                Y_train_pred = self.model(train_x)
                Y_train_pred = Y_train_pred.detach().numpy()

                if self.is_y_pca:
                    Y_train_pred = self.pca_.inverse_transform(Y_train_pred)

                train_mse = mean_squared_error(self.train_y, Y_train_pred)
                train_mse_arr.append(train_mse)

                Y_test_pred = self.model(test_x)
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
                self.model(as_tensor(self.train_x)).detach().numpy())
            self.pred_test_output_ = self.pca_.inverse_transform(
                self.model(as_tensor(self.test_x)).detach().numpy())
        else:
            self.pred_train_output_ = self.model(
                as_tensor(self.train_x)).detach().numpy()
            self.pred_test_output_ = self.model(
                as_tensor(self.test_x)).detach().numpy()
        

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
                self.model(as_tensor(new_data_x)).detach().numpy())
        else:
            self.pred_new_data_output_ = self.model(
                as_tensor(new_data_x)).detach().numpy()

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
        # average over the rows
        self.rmse_train_ = np.sqrt(np.mean(squerr, axis=0))
    
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
        # average over the rows
        self.rmse_test_ = np.sqrt(np.mean(squerr, axis=0))
    
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
        # average over the rows
        self.rmse_new_data_ = np.sqrt(np.mean(squerr, axis=0))
        
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