# simple-neural-network.py

import numpy

training_set_inputs = numpy.array(
    [
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
    ])
training_set_outputs = numpy.array([[0, 1, 1, 0]]).T

numpy.random.seed(1)
synaptic_weights = 2 * numpy.random.random((3, 1)) -1
for i in range(10000):
    output = 1 / (1 + numpy.exp(
        -(numpy.dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += numpy.dot(
        training_set_inputs.T,
        (training_set_outputs - output) * output * (1 - output))

print(1 / 1 + numpy.exp(
    -(numpy.dot(
        numpy.array([1, 0, 0]), synaptic_weights))))
