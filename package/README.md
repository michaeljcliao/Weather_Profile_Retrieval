# MIMORegressionMachineLearner: Machine Learning for Multi-Input-Multi-Output Tasks

## Author
Je-Ching Liao | Academia Sinica | University of Michigan

## Date
June 2024

## Overview
This Python package provides tools to apply machine learning to multi-input-multi-output (MIMO) regression tasks using Random Forest, XGBoost, or Neural Networks. The package is designed to handle various preprocessing steps, model training, and evaluation, making it easier to implement complex regression models.

## Installation
To use this package, ensure you have the following dependencies installed:

- Python 3.6+
- NumPy
- Scikit-learn
- XGBoost
- PyTorch
- Matplotlib

You can install these dependencies using pip:
```bash
pip install numpy scikit-learn xgboost torch matplotlib
```

## Usage

### Importing the Package
```python
from rcecML import MIMORegressionMachineLearner
```

### Initializing the Learner
```python
learner = MIMORegressionMachineLearner()
```

### Loading Data
```python
learner.dataLoader('path/to/x_data.csv', 'path/to/y_data.csv', test_size=0.2)
```
Tip: Before loading data, you can set the current directory in which the desired dataset files could be found by `os.chdir()`. This way, long file paths as the arguments can be avoided. Make sure you `import os` first.

### Data Preprocessing
#### Standardize Input Data
```python
learner.standardizeX()
```

#### Apply PCA to Output Data
```python
learner.pcaY(n_components=5)
```
Tip: Don't forget to specify the number of principal components.

### Model Initialization
#### Random Forest
```python
learner.randomForestRegressorInitialize()
```
Note: The default Random Forest algorithm comes with the default hyperparameters of `n_estimators` set to 100, `random_state` set to match the specified random state, `criterion` set to 'squared_error', `max_depth` set to None, and `min_samples_split` set to 2.

Tip: Modify the hyperparameters with `hyperparamSetter()`. Refer to [`sklearn.ensemble.RandomForestRegressor()` documentations](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

#### XGBoost
```python
learner.XGBRegressorInitialize()
```
Note: The default XGBoost algorithm comes with the default hyperparameters of `random_state` set to match the specified random state, `tree_method` set to 'hist', and `multi_strategy` set to 'multi_output_tree'.

Tip: Modify the hyperparameters with `hyperparamSetter()`. Refer to [`xgboost.sklearn.XGBRegressor()` parameters documentations](https://xgboost.readthedocs.io/en/stable/parameter.html).

#### Neural Network
```python
learner.neuralNetworkInitialize()
```
Note: The Neural Network algorithm comes with 6 hyperparameters, `clip_value`, `isClipped`, `neuron_list`, `activation_list`, `criterion`, and `optimizer`. Feel free to customize. Description to these hyperparameters are as below:
- **`clip_value`**: *(default: 5)*  
  The value to which gradients will be clipped. Gradient clipping helps in preventing the problem of exploding gradients in neural networks.

- **`isClipped`**: *(default: True)*  
  A boolean flag indicating whether gradient clipping should be applied. If `True`, gradients will be clipped to the `clip_value`. If `False`, gradient clipping will not be applied.

- **`neuron_list`**: *(default: None)*  
  A list specifying the number of neurons in each layer of the neural network. Each entry in the list represents a layer, and the value at each entry represents the number of neurons in that layer. The default is `[x_input_dim, 30, 20, y_output-dim]`, i.e., 2 hidden layers + 1 output layer.
  Tip: This is alter with you add, remove or modify a layer.

- **`activation_list`**: *(default: None)*  
  A list of activation functions for each layer in the neural network. Each entry in the list represents a layer, and the value at each entry specifies the activation function to be used for that layer. The default is `[nn.ReLU(), nn.ReLU()]`, accompanying the 2-hidden-layer default setting.
  Tip: Refer to [Pytorch nn documentation (Non-linear Activations)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) for the available activation functions and their customization.

- **`criterion`**: *(default: None)*  
  The loss function used to train the neural network. If not specified, a default loss function will be used. The default is `nn.MSELoss(reduction='sum')`.
  Tip: Refer to [Pytorch nn documentation (Non-linear Activations)](https://pytorch.org/docs/stable/nn.html#loss-functions) for the available loss functions and their customization.

- **`optimizer`**: *(default: None)*  
  The optimization algorithm used to update the weights of the neural network. If not specified, a default optimizer will be used. The default is `optim.AdamW(self.model.parameters(), lr=0.01)`.
  Tip: Refer to [Pytorch nn documentation (Non-linear Activations)](https://pytorch.org/docs/stable/optim.html) for the available optimization algorithms and their customization. Make sure the `parameters()` of the model is passed into the optimizer method.

### Model Training
#### Random Forest or XGBoost
```python
learner.trainRFXGB()
```

#### Neural Network
```python
learner.trainNN(epochs=500)
```
Tip: Don't forget to specify the number of epochs the model will be trained over.

### Predictions
#### Predict with Random Forest or XGBoost
```python
learner.predictRFXGB()
```
Note: Both the training and test x data are fed into the trained model to produce two y prediction data sets. They are saved as class attributes.

#### Predict with Neural Network
```python
learner.predictNN()
```
Note: Both the training and test x data are fed into the trained model to produce two y prediction data sets. They are saved as class attributes.

#### Predict New Data with Random Forest or XGBoost
```python
learner.predictNewDataRFXGB(new_data_x)
```
Note: User-provided x data is fed into this method to produce a y prediction data set. It is saved as a class attribute.

#### Predict with Neural Network
```python
learner.predictNewDataNN(new_data_x)
```
Note: User-provided x data is fed into this method to produce a y prediction data set. It is saved as a class attribute.

### Evaluation
#### RMSE Calculation
```python
learner.rmseOverTrain()
learner.rmseOverTest()
learner.rmseOverNewData(new_data_y)

```
Note: This method calculate the Root Mean Squared Error (RMSE) across all rows of prediction against the ground truth. `isPercentage` hyperparameter determines whether the RMSE is in ratio or in difference. isPercentage being True means that the RMSE will be calculated as (pred - truth)/truth, while isPercentage being False means that the RMSE is calculated as (pred - truth). They are saved as class attributes.
Tip: User-provided y data that corresponds to the x data fed into `predictWithNewData()` is required.

#### Plotting Neural Network Loss
```python
learner.plotLossNN()
```

#### Plotting Mean Squared Error for Neural Network
```python
learner.plotMSENN()
```

## Customization
### Set Hyperparameters
#### Random Forest
```python
learner.hyperparamSetter('n_estimators', 200)
```
Note: If Random Forest is initialized beforehand, calling this method will modify the hyperparameter set of the Random Forest model.

#### XGBoost
```python
learner.hyperparamSetter('max_depth', 6)
```
Note: If XGBoost is initialized beforehand, calling this method will modify the hyperparameter set of the XGBoost model.

### Modify Neural Network Architecture
#### Add Layer
```python
learner.neuralNetworkAddLayer(prev_layer=2, out_neurons=40)
```
Note: the `prev_layer` is the layer number (starting with 1) after which the new layer will be inserted.
Tip: One and only one of the input/output dimensions of the to-be-added layer must be specified. The activation function for this layer will by default be `nn.ReLU()` unless assigned when calling this method. Refer to [Pytorch nn documentation (Non-linear Activations)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) for the available activation functions and their customization.

#### Remove Layer
```python
learner.neuralNetworkRemoveLayer(num_layer=3)
```
Note: the `num_layer` is the layer number (starting with 1) to be removed.

#### Modify Neurons in Layer
```python
learner.neuralNetworkModifyNeuron(num_layer=2, in_neurons=50)
```
Note: the `num_layer` is the layer number (starting with 1) to be modified.
Tip: At least one new value for the input/output dimensions has to be specified.

#### Change Activation Function of a Layer
```python
learner.neuralNetworkModifyActivation(num_layer=2, activation_function=torch.nn.Tanh())
```
Note: the `num_layer` is the layer number (starting with 1) to be modified.
Tip: The new activation function must be specified. Refer to [Pytorch nn documentation (Non-linear Activations)](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) for the available activation functions and their customization.

### Visualize Neural Network Architecture
```python
learner.neuralNetworkVisualizer()
```
Note: You can visualize the architecture of the Neural Network model with this method.

## Class Attributes
#### Note: attributes that are learned or are results from the machine learning is suffixed by a "_"
### Initialization Attributes

- `random_state`: The random state for reproducibility.
- `is_y_pca`: Flag indicating whether PCA was applied to the output labels.
- `x_input_dim`: Dimension of the input features.
- `y_output_dim`: Dimension of the output labels.
- `isModelTrained`: Flag indicating whether the model has been trained.
- `model_type`: The type of model being used.
- `rf_grid`: Default hyperparameters for Random Forest.
- `xgb_grid`: Default hyperparameters for XGBoost.
- `nn_neuron_list`: List of neurons for each layer in the Neural Network.
- `nn_activation_list`: List of activation functions for each layer in the Neural Network.
- `neuralNet`: Neural Network model.

### Data Attributes

- `train_x`: Training input features.
- `test_x`: Testing input features.
- `train_y`: Training output labels.
- `test_y`: Testing output labels.
- `train_y_pca_`: PCA-transformed training output labels.
- `test_y_pca_`: PCA-transformed testing output labels.
- `pca_`: PCA model used for transforming the output labels.

### Model Attributes

- `model`: The chosen machine learning model.
- `criterion`: Loss function for the Neural Network.
- `optimizer`: Optimizer for the Neural Network.
- `clip_value`: Value for gradient clipping in the Neural Network.
- `isClipped`: Flag indicating whether gradient clipping is enabled.

### Training Attributes

- `training_time_`: Time taken to train the model.
- `loss_array_`: Array of loss values during Neural Network training.
- `train_mse_array_`: Array of training Mean Squared Error (MSE) values during Neural Network training.
- `test_mse_array_`: Array of testing Mean Squared Error (MSE) values during Neural Network training.

### Prediction Attributes

- `pred_train_output_`: Predicted output for the training data.
- `pred_test_output_`: Predicted output for the testing data.
- `pred_new_data_output_`: Predicted output for new input data.
- `train_mse_loss_`: Mean Squared Error for the training data.
- `test_mse_loss_`: Mean Squared Error for the testing data.
- `rmse_train_`: RMSE for the training data.
- `rmse_test_`: RMSE for the testing data.
- `rmse_new_data_`: RMSE for new input data.

## Functions

- `refreshNNNeuronList()`: Updates the list of neurons for the Neural Network based on input and output dimensions.
- `dataLoader(var_x, var_y, test_size)`: Loads and splits the data.
- `standardizeX()`: Standardizes the input features.
- `pcaY(n_components)`: Applies PCA to the output labels.
- `hyperparamSetter(param_name, param_val)`: Sets hyperparameters for the chosen model.
- `randomForestRegressorInitialize()`: Initializes the Random Forest model.
- `XGBRegressorInitialize()`: Initializes the XGBoost model.
- `neuralNetworkInitialize()`: Initializes the Neural Network model.
- `neuralNetworkCriterionSetter(criterion)`: Sets the loss function for the Neural Network.
- `neuralNetworkOptimizerSetter(optimizer)`: Sets the optimizer for the Neural Network.
- `neuralNetworkAddLayer(prev_layer, in_neurons, out_neurons, activation_function)`: Adds a layer to the Neural Network.
- `neuralNetworkRemoveLayer(num_layer, isPrevPreserved)`: Removes a layer from the Neural Network.
- `neuralNetworkModifyNeuron(num_layer, in_neurons, out_neurons)`: Modifies the number of neurons in a layer of the Neural Network.
- `neuralNetworkModifyActivation(num_layer, activation_function)`: Modifies the activation function of a layer in the Neural Network.
- `neuralNetworkVisualizer()`: Visualizes the structure of the Neural Network.
- `trainRFXGB()`: Trains the Random Forest or XGBoost model.
- `trainNN(epochs)`: Trains the Neural Network.
- `predictRFXGB()`: Makes predictions using the trained Random Forest or XGBoost model.
- `predictNN()`: Makes predictions using the trained Neural Network.
- `predictNewDataRFXGB(new_data_x)`: Makes predictions on new data using the trained Random Forest or XGBoost model.
- `predictNewDataNN(new_data_x)`: Makes predictions on new data using the trained Neural Network.
- `rmseOverTrain(isPercentage)`: Calculates RMSE over the training data.
- `rmseOverTest(isPercentage)`: Calculates RMSE over the testing data.
- `rmseOverNewData(new_data_y, isPercentage)`: Calculates RMSE over new data.
- `plotLossNN(xlabel, ylabel, title)`: Plots the loss over training epochs for the Neural Network.
- `plotMSENN(xlabel, ylabel, title)`: Plots the Mean Squared Error over training epochs for the Neural Network.

## Notes
- Ensure to initialize the model before setting hyperparameters.
- Always re-initialize the neural network model after modifying its architecture.
- For training the neural network, it is essential to specify the number of epochs.

## License
This project is licensed under the MIT License.

## Contact
For any questions or issues, please contact Je-Ching Liao at jechingliao@gmail.com / michaeljcliao@gmail.com .

Enjoy using the MIMORegressionMachineLearner package!