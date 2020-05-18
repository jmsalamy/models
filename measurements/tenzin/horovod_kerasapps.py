from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
import timeit

import tensorflow as tf
import horovod.tensorflow.keras as hvd
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

hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.allow_growth = False
    config.gpu_options.visible_device_list = ''
K.set_session(tf.Session(config=config))

# Set up standard model.
#model = keras.applications.vgg16.VGG16(weights=None, input_shape=(32, 32, 3))
#model = keras.applications.resnet50.ResNet50(weights=None, input_shape=(32, 32, 3), include_top=True, classes=10)
#model = keras.applications.VGG19(weights=None, input_shape=(32, 32, 3))
model = keras.applications.NASNetLarge(weights=None, input_shape=(32, 32, 3))
#model = keras.applications.NASNetMobile(weights=None, input_shape=(32, 32, 3))

opt = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt, compression=compression)

model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['accuracy'])

hooks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
#x_train, x_test = x_train / 255.0, x_test / 255.0

def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

log('Model: simple cnn')
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    model.summary()
    #checkpoint_path = './checkpoints/checkpoint-{epoch}.h5'
    #model.save(checkpoint_path.format(epoch=0))
    #hooks.append(keras.callbacks.ModelCheckpoint(checkpoint_path))

def train():
    model.fit(x_train[0:args.batch_size], y_train[0:args.batch_size],
          batch_size=args.batch_size,
          callbacks=hooks,
          epochs=args.num_epochs,
          steps_per_epoch=args.num_iters,
          validation_data=(x_test, y_test),
          verbose=2 if hvd.rank() == 0 else 0)

log('Train time: %f' % timeit.timeit(train, number=1))

score = model.evaluate(x_test, y_test, verbose=0)
log('Test loss: %f' % score[0])
log('Test accuracy: %f' % score[1])
