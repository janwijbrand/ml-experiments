# following along with this article:
#  https://realpython.com/python-ai-neural-network/

import numpy
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
        self.weights = numpy.array(
            [numpy.random.randn(), numpy.random.randn()])
        self.bias = numpy.random.randn()

    def _sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def _sigmoid_deriv(self, x):
        sigmoid_x = self._sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def _compute_gradients(self, input_vector, target):
        prediction, layer2, layer1 = self.predict(input_vector)
        derror_dprediction = 2 * (prediction - target)
        dprediction_layer1 = self._sigmoid_deriv(layer1)
        dlayer1_dbias = 1
        # WTF?        ------+
        #                   V
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        derror_dbias = (
            derror_dprediction * dprediction_layer1 * dlayer1_dbias
        )
        derror_weights = (
            derror_dprediction * dprediction_layer1 * dlayer1_dweights
        )
        return derror_dbias, derror_weights

    def _update_parameter(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            random_data_index = numpy.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target)
            self._update_parameter(derror_dbias, derror_dweights)
            if current_iteration % 100 == 0:
                cumulative_error = 0
                for input_vector, target in zip(input_vectors, targets):
                    prediction = self.predict(input_vector)
                    error = numpy.square(prediction - target)
                    cumulative_error += error
                cumulative_errors.append(cumulative_error)
        return cumulative_errors

    def predict(self, input_vector):
        layer1 = numpy.dot(input_vector, self.weights) + self.bias
        layer2 = self._sigmoid(layer1)
        prediction = layer2
        return prediction, layer2, layer1


n1 = NeuralNetwork(0.1)
print(n1.predict(numpy.array([1.66, 1.56])))
print(n1.predict(numpy.array([2, 1.5])))

input_vectors = numpy.array([
    [3, 1.5],
    [2, 1],
    [4, 1.5],
    [3, 4],
    [3.5, 0.5],
    [2, 0.5],
    [5.5, 1],
    [1, 1],
])
targets = numpy.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1
n2 = NeuralNetwork(learning_rate)
training_error = n2.train(input_vectors, targets, 10000)
plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.show()

print(n2.predict(numpy.array([1.66, 1.56])))
print(n2.predict(numpy.array([2, 1.5])))
