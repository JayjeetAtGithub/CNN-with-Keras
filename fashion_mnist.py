import imageio
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Collects the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()


# Labels of the datasets
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Details of the training images
# There are 60,000 images each represented as 28x28 pixels
print(train_images.shape)
# Similarly, there are 60,000 training labels
print(len(train_labels))


# Preprocess data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# We can see that the pixel values lie in the range of 0 to 255
# So we divide it with 255.0 to bring the pixels in range 0 and 1

train_images = train_images / 255.0
test_images = test_images / 255.0


# Check the first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Define the neural net
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)


# Test the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# Making predictions
predictions = model.predict(test_images)
print(np.argmax(predictions[7797]))

# Plot test image 7797 with matplotlib
plt.figure()
plt.imshow(test_images[7797])
plt.colorbar()
plt.grid(False)
plt.show()
