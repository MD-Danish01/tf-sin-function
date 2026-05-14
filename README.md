# Sine Wave Prediction using TensorFlow/Keras

A beginner-friendly deep learning project that trains a neural network to learn and predict sine wave values using Python, NumPy, TensorFlow, and Keras.

This project was created to understand the basic workflow of regression-based neural network training, model evaluation, prediction visualization, and TensorFlow Lite model conversion.

---

## Project Overview

In this project, a neural network is trained to approximate a sine wave.

The model takes an input value `x` and predicts the corresponding sine value `sin(x)`.

To make the task more realistic, random noise is added to the sine wave data before training.

---

## What This Project Does

- Generates random input values between `0` and `2π`
- Creates noisy sine wave output values
- Splits the data into training, validation, and testing sets
- Builds a small neural network using TensorFlow/Keras
- Trains the model to predict sine wave values
- Visualizes training and validation loss
- Compares predicted values with actual sine wave values
- Converts the trained Keras model into TensorFlow Lite format
- Converts the TensorFlow Lite model into a C header file

---

## Tech Stack

- Python
- Google Colab / Jupyter Notebook
- NumPy
- Matplotlib
- TensorFlow
- Keras
- TensorFlow Lite

---

## Dataset

The dataset is generated manually using NumPy.

```python
x_values = np.random.uniform(low=0, high=2 * math.pi, size=1000)
y_values = np.sin(x_values) + (0.1 * np.random.randn(x_values.shape[0]))
```

## Dataset details:

Total samples: 1000
Input: Random values between 0 and 2π
Output: Noisy sine wave values
Training data: 60%
Validation data: 20%
Testing data: 20%
Model Architecture

The model is a small feed-forward neural network.

model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

## Model summary:

Layer	Neurons	Activation
Dense Layer 1	16	ReLU
Dense Layer 2	16	ReLU
Output Layer	1	Linear

Total parameters: 321

Model Compilation

The model is compiled using:
```
model.compile(
    optimizer='rmsprop',
    loss='mae',
    metrics=['mae']
)
```

## Loss function used:

Mean Absolute Error

## Optimizer used:

RMSprop
Training

The model is trained for 500 epochs.
```
history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=100,
    validation_data=(x_val, y_val)
)
```
During training, both training loss and validation loss are tracked.

## Results

After training, the model learns to approximate the sine wave pattern.

Final validation result from the notebook:
```
Validation MAE: around 0.0924
```
The model predictions are plotted against the actual test values to visually compare performance.
