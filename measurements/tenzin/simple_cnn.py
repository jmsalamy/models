from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
from tensorflow import keras

import fake_data

parser = argparse.ArgumentParser(description='Simple CNN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--num-epochs', type=int, default=1,
                    help='number of epochs')
parser.add_argument('--num-iters', type=int, default=None,
                    help='number of batch iterations')

args = parser.parse_args()

#(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
(train_images, train_labels), (test_images, test_labels) = fake_data.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Set up standard model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

#opt = keras.optimizers.Adam(0.001)
opt = keras.optimizers.SGD

#model.compile(optimizer=opt,
#              loss="sparse_categorical_crossentropy",
#              metrics=['accuracy'])
model.compile(optimizer=opt,
              loss="mean_squared_error",
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=args.num_epochs, steps_per_epoch=args.num_iters,
                    batch_size=args.batch_size, verbose=2, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
model.save('simple_cnn_model.h5')
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
