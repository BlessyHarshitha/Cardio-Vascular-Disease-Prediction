# Heart Disease Prediction using Neural Networks

## Overview

This project implements various neural network architectures, including Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Bidirectional RNNs (BiRNN), to predict the presence of heart disease based on a dataset. The models are trained and evaluated, and an ensemble approach is used to combine predictions from multiple models for improved accuracy.

## Requirements

To run this project, you need the following libraries:

- pandas
- numpy
- scikit-learn
- tensorflow

You can install these libraries using pip:

bash
pip install pandas numpy scikit-learn tensorflow

# Heart Disease Prediction using Neural Networks

## Overview

This project implements various neural network architectures, including Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Bidirectional RNNs (BiRNN), to predict the presence of heart disease based on a dataset. The models are trained and evaluated, and an ensemble approach is used to combine predictions from multiple models for improved accuracy.

## Dataset

The dataset used in this project is heart disease.csv, which contains various features related to heart health along with a target label indicating the presence or absence of heart disease.

## Steps

### 1. Data Loading and Preprocessing
- Load the dataset using Pandas.
- Normalize the features using StandardScaler.
- Split the data into training and testing sets.

### 2. Model Development
Four different models are created:
- *Deep Neural Network (DNN)*: A simple feedforward network.
- *Convolutional Neural Network (CNN)*: A model designed for handling grid-like data, reshaped to fit 1D convolution.
- *Recurrent Neural Network (RNN)*: A model suitable for sequential data.
- *Bidirectional RNN (BiRNN)*: An RNN that processes data in both forward and backward directions.

### 3. Ensemble Model Training and Prediction
- Each model is trained on the training data.
- Predictions are made on the test data.

### 4. Ensemble Predictions
- The predictions from all models are averaged to form an ensemble prediction.

### 5. Evaluation
- The ensemble predictions are evaluated using metrics such as accuracy, precision, recall, and F1-score.


## Results

After evaluating the ensemble model, the following metrics were obtained:

- *Accuracy*: (value)
- *Precision*: (value)
- *Recall*: (value)
- *F1-Score*: (value)

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- Thanks to the contributors of the dataset.
- Special thanks to the TensorFlow team for their powerful machine learning framework.
