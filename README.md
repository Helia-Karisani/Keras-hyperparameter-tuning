
# Keras Hyperparameter Tuning on MNIST


## Overview

This project demonstrates hyperparameter tuning in Keras using Keras Tuner.

The notebook trains a simple neural network classifier on the MNIST handwritten digit dataset. Instead of manually choosing the number of hidden units and learning rate, it uses `keras-tuner` to search for better hyperparameter values.

The project shows the full workflow:

1. install and import required packages
2. load and preprocess MNIST
3. define a model-building function
4. create a hyperparameter search tuner
5. run hyperparameter search
6. retrieve the best hyperparameters
7. rebuild and train the best model
8. evaluate the final model

## What the Project Does Step by Step

### 1. Environment Setup

The notebook installs and checks:

- TensorFlow
- NumPy
- Keras Tuner

It also increases the Python recursion limit and suppresses unnecessary warnings/log messages.

### 2. Load and Pre-process MNIST Dataset

load and pre-process MNIST dataset

The notebook loads MNIST using:

- `mnist.load_data()`

Then it normalizes pixel values:

- original pixel range: `[0, 255]`
- normalized range: `[0, 1]`

The data shapes are:

- training data: `(60000, 28, 28)`
- validation/test data: `(10000, 28, 28)`

This prepares the image data for neural network training.

## Model Structure

The model is a simple feedforward neural network.

Architecture:

- `Flatten(input_shape=(28, 28))`
- `Dense(units=hp.Int(...), activation='relu')`
- `Dense(10, activation='softmax')`

The first layer flattens each `28 x 28` image into a 784-dimensional vector.  
The hidden dense layer learns features from the image.  
The final dense layer outputs probabilities for the 10 digit classes.

## Hyperparameter Tuning

define a function that returns a compiled model that is ready for hyperparameter tuning.

This function builds and compiles a Keras model with hyperparameters:

- `hp.Int('units', ...)`: Defines the number of units in the Dense layer as a hyperparameter.
- `hp.Float('learning_rate', ...)`: Defines the learning rate as a hyperparameter.
- `model.compile()`: Compiles the model with the Adam optimizer and sparse categorical cross-entropy loss.

### Tuned Hyperparameters

The notebook tunes two hyperparameters:

- number of units in the hidden Dense layer
- learning rate for the Adam optimizer

Search space:

- `units`: from `32` to `512`, step size `32`
- `learning_rate`: from `1e-4` to `1e-2`, sampled logarithmically

## Random Search Tuner

### creating Hyperparameter search tuner

Here we create a RandomSearch tuner and specify the model-building function, optimization objective, number of trials, and directory for storing results.

- `build_model`: The model-building function.
- `objective='val_accuracy'`: The metric to optimize.
- `max_trials=10`: The maximum number of different hyperparameter configurations to try.
- `executions_per_trial=2`: The number of times to run each configuration.
- `directory='my_dir'`: Directory to save the results.
- `project_name='intro_to_kt'`: Name of the project for organizing results.
- Epoch = one full pass over the training data.
- Trial = one full training experiment with a specific setting.

## Running the Hyperparameter Search

### Running the hyperparameter search

Pass in the training data, validation data, and the number of epochs.

The tuner tests multiple combinations of hidden-layer size and learning rate. Each trial trains the model for 5 epochs and evaluates validation accuracy.

Best result from the search:

- best validation accuracy: about `0.9804`
- best number of units: `512`
- best learning rate: about `0.00152`

## Using the Best Hyperparameters

using it

After the search, the notebook retrieves the best hyperparameters and builds a new model with them.

Best selected values:

- units: `512`
- learning rate: `0.0015226984419050577`

Then the best model is trained for 10 epochs using a validation split.

Final reported test accuracy:

- `0.9765`

## Why the Optimizer and Loss Function Were Chosen

### Optimizer: Adam

The notebook uses the Adam optimizer because it is a strong default optimizer for neural networks. It adapts the learning rate during training and usually converges faster and more reliably than plain stochastic gradient descent.

In this project, the learning rate is tuned because it strongly affects training speed and final accuracy. A learning rate that is too small trains slowly, while a learning rate that is too large can make training unstable.

### Loss: Sparse Categorical Cross-Entropy

The notebook uses `sparse_categorical_crossentropy` because MNIST is a multi-class classification problem with 10 classes.

The labels are integer class IDs from `0` to `9`, not one-hot encoded vectors. Therefore, sparse categorical cross-entropy is the correct loss function.

The model output uses `softmax`, so the loss compares predicted class probabilities against the true integer labels.

## Result Analysis

The hyperparameter search found that a large hidden layer with `512` units and a learning rate around `0.0015` worked best among the tested settings.

The final model reached about `97.65%` test accuracy, which is a strong result for a simple dense neural network on MNIST.

The training accuracy increased up to about `99.6%`, while validation accuracy stayed around `97.6%`. This means the model learned the training data very well, but there is some overfitting because the training accuracy is noticeably higher than the validation accuracy.

The validation loss also increased near the later epochs, even while training loss kept decreasing. This suggests that training for fewer epochs or using early stopping could improve generalization.

## Technical Characteristics

- hyperparameter tuning with Keras Tuner
- random search over model settings
- dynamic model-building function using `hp`
- tuning hidden-layer width
- tuning Adam learning rate
- validation accuracy as the search objective
- repeated executions per trial
- final retraining using the best hyperparameters
- MNIST multi-class image classification

## Packages Used

- `tensorflow`
- `numpy`
- `keras_tuner`
- `os`
- `warnings`
- `sys`

Main Keras components used:

- `tensorflow.keras.datasets.mnist`
- `tensorflow.keras.models.Sequential`
- `tensorflow.keras.layers.Flatten`
- `tensorflow.keras.layers.Dense`
- `tensorflow.keras.optimizers.Adam`

Keras Tuner component used:

- `keras_tuner.RandomSearch`

## Files

- `Keras- hyperparameter-tuning.ipynb`
- `README.md`


The `my_dir/intro_to_kt/` folder stores Keras Tuner search results. It is generated automatically during tuning and does not need to be manually edited.

## Summary

This project shows how to use Keras Tuner to automatically search for good hyperparameters for an MNIST classifier. It tunes the number of units in the hidden layer and the Adam learning rate, selects the best configuration based on validation accuracy, retrains the best model, and evaluates it on the test set. The final model performs well, reaching about `97.65%` accuracy, but the training logs also show mild overfitting in later epochs.

