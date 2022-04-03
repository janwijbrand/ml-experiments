# following along with this article:
#  https://realpython.com/python-ai-neural-network/

from collections import Counter
from numpy import dot, array, exp, square

weights = [1.45, -0.66]
bias = array([0.0])
fraction = 1
training_set = [
    [array([1.66, 1.56]), 1],
    [array([2.00, 1.50]), 0],
]
iterations = Counter()
max_iterations = 10000

def sigmoid(x):
    iterations['signmoid'] += 1
    return 1 / (1 + exp(-x))

def predict(input_vector, weights, bias):
    iterations['predictions'] += 1
    layer_1: float = dot(input_vector, weights) + bias
    print('layer_1 result', layer_1)
    return layer_1
    layer_2: float = sigmoid(layer_1)
    print('layer_2 result', layer_2)
    return layer_2

def adjust(weights, prediction, target):
    iterations['adjustments'] += 1
    error = square(prediction - target)
    derivative = 2 * (prediction - target)
    adjusted_weights = weights - (derivative * fraction)
    print(
        'error', error,
        'derivative', derivative,
        'fraction', fraction,
        'adjustment', adjusted_weights)
    return adjusted_weights

correct = False
iterations['iterations']
while not correct and iterations['iterations'] < max_iterations:
    print('====')
    iterations['iterations'] += 1
    for input, target in training_set:
        print('training', input, 'should be', target)
        prediction = predict(input, weights, bias)
        if round(prediction[0]) == target:
            correct = True
            print('correct prediction')
        else:
            correct = False
            print('incorrect prediction')
            weights = adjust(weights, prediction, target)

print('====')
print(iterations)
