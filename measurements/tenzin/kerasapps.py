from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
import timeit

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# Benchmark settings
parser = argparse.ArgumentParser(description='Horovod Simple CNN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--num-epochs', type=int, default=1,
                    help='number of epochs')
parser.add_argument('--num-iters', type=int, default=None,
                    help='number of batch iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda

config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.allow_growth = False
    config.gpu_options.visible_device_list = ''
K.set_session(tf.Session(config=config))

# Set up standard model.
model = keras.applications.vgg16.VGG16(weights=None, input_shape=(32, 32, 3))
#model = keras.applications.resnet50.ResNet50(weights=None, input_shape=(32, 32, 3))
#model = keras.applications.VGG19(weights=None, input_shape=(32, 32, 3))

opt = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

model.summary()

def train():
    model.fit(x_train[0:args.batch_size], y_train[0:args.batch_size],
          batch_size=args.batch_size,
          epochs=args.num_epochs,
          steps_per_epoch=args.num_iters,
          validation_data=(x_test, y_test),
          verbose=2)

print('Train time: %f' % timeit.timeit(train, number=1))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: %f' % score[0])
print('Test accuracy: %f' % score[1])
