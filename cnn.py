from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.models import Sequential
"""
Convolutional Neural Network to classify images of cats and dogs
Convolution -> Max Pooling -> Flattening -> Full Connection
"""

# Check if GPU is available or not
import tensorflow as tf
print(tf.test.is_gpu_available())

# Define a Sequential Neural Network
classifier = Sequential()

# Convolution
# 32 feature detectors , 3 by 3 feature detectors
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Max Pooling
# pool size is 2 by 2
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution 2nd Laye
classifier.add(Conv2D(32, (3, 3), activation='relu'))

# Max Pooling 2nd Layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Flattening
classifier.add(Flatten())

# Full Connection / Fully Connected Layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#  Fitting the CNN to our images and training our model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# Since we have 2 classes 'cats' and 'dogs', classmode is binary
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=test_set,
    validation_steps=2000
)

# Making new predictions

test_image = image.load_img(
    'dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    print('dog')
else:
    print('cat')
