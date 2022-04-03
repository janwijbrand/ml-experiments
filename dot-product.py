# following along with this article:
#  https://realpython.com/python-ai-neural-network/

from numpy import dot

input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]
weights_3 = [-2.34, -9.1]

print(
    dot(
        input_vector,
        weights_1))

print(
    dot(
        input_vector,
        weights_2))

print(
    dot(
        input_vector,
        weights_3))

print(
    dot(
        input_vector,
        input_vector))

