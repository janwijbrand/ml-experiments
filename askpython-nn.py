# * https://www.askpython.com/python/examples/neural-networks

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

train_images = train_images / 255
test_images = test_images / 255

print("First Label before conversion:")
print(train_labels[0])

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

print("First Label after conversion:")
print(train_labels[0])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    x=train_images,
    y=train_labels,
    epochs=10
)

plt.plot(history.history['loss'], color='blue')
plt.plot(history.history['accuracy'], color='orange')
plt.xlabel('epochs')
plt.ylabel(['loss', 'accuracy'])
plt.show()

test_loss, test_accuracy = model.evaluate(
    x=test_images,
    y=test_labels
)

print("Test Loss: %.4f" % test_loss)
print("Test Accuracy: %.4f" % test_accuracy)
