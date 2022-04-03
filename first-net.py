# first-net.py
from numpy import dot, array, exp, square

weights_1 = [1.45, -0.66]
bias = array([0.0])

def sigmoid(x):
    return 1 / (1 + exp(-x))

def predict(input_vector, weights, bias):
    layer_1: float = dot(input_vector, weights) + bias
    layer_2: float = sigmoid(layer_1)
    return layer_2

input_vector = array([1.66, 1.56])
prediction = predict(input_vector, weights_1, bias)
print('prediction', prediction, round(prediction[0]), 'should be 1')

input_vector = array([2, 1.5])
prediction = predict(input_vector, weights_1, bias)
print('prediction', prediction, round(prediction[0]), 'should be 0')

target = 0
mse = square(prediction - target)
derivative = 2 * (prediction - target)
print('error', mse, 'derivative', derivative)

fraction = 1
weights_1 = weights_1 - derivative * fraction
prediction = predict(input_vector, weights_1, bias)
print('prediction', prediction, round(prediction[0]), 'should be 0')

def sigmoid_deriv(x):
    # Proof for this:
    #  https://beckernick.github.io/sigmoid-derivative-neural-network/
    return sigmoid(x) * (1 - sigmoid(x))

derror_dprediction = 2 * (prediction - target)
layer_1 = dot(input_vector, weights_1) + bias
dprediction_dlayer1 = sigmoid_deriv(layer_1)
dlayer1_dbias = 1
derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
