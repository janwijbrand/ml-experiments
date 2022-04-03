from numpy import dot, array, exp, square

weights_1 = [1.45, -0.66]
bias = array([0.0])

def sigmoid(x):
    return 1 / (1 + exp(-x))

def predict(input_vector, weights, bias):
    layer_1: float = dot(input_vector, weights) + bias
    layer_2: float = sigmoid(layer_1)
    return layer_2
