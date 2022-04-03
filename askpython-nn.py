import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)

# Displaying first 9 images of dataset
# fig = plt.figure(figsize=(30, 30))
# nrows=9
# ncols=9
# for i in range(27):
#   fig.add_subplot(nrows, ncols, i+1)
#   plt.imshow(train_images[i])
#   plt.title("Digit: {}".format(train_labels[i]))
#   plt.axis(False)
# plt.show()
