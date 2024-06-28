# Author: Je-Ching Liao | Academia Sinica | University of Michigan
# Date: June 2024
# File Purpose: .py file to test the functionality of rcecML.py package, the 
#               package to apply machine learning to multi-input-multi- 
#               output tasks with Random Forest, XGBoost, or Neural Network
# %% ------------------------------------------------------------------------
import os
import torch
from torch import nn
import rcecML
from pytorch_model_summary import summary
# %% ------------------------------------------------------------------------
# %% ------------------------------------------------------------------------
os.chdir('/NFS15/michaelliao/retrievaldata')
# %% ------------------------------------------------------------------------
# random_seed = 7036
n_components = 5
# %% ------------------------------------------------------------------------
ML = rcecML.MIMORegressionMachineLearner()
# %% ------------------------------------------------------------------------
ML.dataLoader(var_x='./train_x.csv',
              var_y='./train_y_Temp.csv',
              test_size=0.2)
# %% ------------------------------------------------------------------------
ML.standardizeX()
# %% ------------------------------------------------------------------------
ML.pcaY(n_components)
# %% ------------------------------------------------------------------------
ML.neuralNetworkInitialize()
ML.neuralNetworkVisualizer()
# %% ------------------------------------------------------------------------
ML.neuralNetworkModifyNeuron(1, out_neurons=33)
ML.neuralNetworkModifyNeuron(2, out_neurons=18)
# ML.neuralNetworkModifyActivation(1, nn.Sigmoid())
ML.neuralNetworkAddLayer(1, out_neurons=25)
ML.neuralNetworkRemoveLayer(3)
# %% ------------------------------------------------------------------------
ML.neuralNetworkVisualizer()
# %% ------------------------------------------------------------------------
ML.trainNN(7000)
# %% ------------------------------------------------------------------------
ML.predictNN()
# %% ------------------------------------------------------------------------
ML.plotLossNN()
ML.plotMSENN()


# %% ------------------------------------------------------------------------
ML.randomForestRegressorInitialize()
# %% ------------------------------------------------------------------------
# ML.hyperparamSetter('min_samples_split', 2)
# %% ------------------------------------------------------------------------
ML.trainRFXGB()
# %% ------------------------------------------------------------------------
ML.predictRFXGB()
