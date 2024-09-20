# Temperature and Vapor Density Profile Retrieval with Spectrometer Data Using Multi-Input-Multi-Output Regression

## Author
Je-Ching (Michael) Liao | Academia Sinica | University of Michigan

## Date
June 2024

## Overview
This project consists of two parts: first part being the developing and experimentation and the second part is the Python package creation. This project uses weather data collected by spectrometer at the ground level, such as the energy level of different radio channels, the ground temperature and humidity, etc, to retrieve the entire temperature and vapor density profiles of the air column above. This project uses the true temperature and vapor density profile data recorded by the radiosonde attached under weather balloons as training data. Recognizing that this is a multi-input-multi-output (MIMO) regression tasks, this project also created a Python package to provides tools to apply MIMO regression tasks using Random Forest, XGBoost, or Neural Networks. The package is designed to handle various preprocessing steps, model training, and evaluation, making it easier to implement complex regression models.

There is a separate README.md for the Python package.

## License
This project is licensed under the MIT License.

## Contact
For any questions or issues, please contact Je-Ching (Michael) Liao at jechingliao@gmail.com / michaeljcliao@gmail.com .
