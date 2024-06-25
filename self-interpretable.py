import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import tensorflow as tf
from keras import backend as K
from skimage.util.shape import view_as_windows
import glob
from os import walk

import argparse
parser = argparse.ArgumentParser()

# Adding argparse arguments
parser.add_argument("--n", type=int, default=4,
                    help="Number of iterations")
parser.add_argument("--randomrestart", type=int, default=1,
                    help="Number of random restarts")
parser.add_argument("--data_path", type=str, required=True,
                    help="Path to the directory containing data files")
parser.add_argument("--best_blackbox_ckpt", type=str, required=True,
                    help="Path to the blackbox classifier's best checkpoint file")
parser.add_argument("--best_interpretable_ckpt", type=str, required=True,
                    help="Path to save checkpoint files")

args = parser.parse_args()
total1 = args.n
randomrestart1 = args.randomrestart
data_path = args.data_path
best_blackbox_ckpt = args.best_blackbox_ckpt
best_interpretable_ckpt = args.best_interpretable_ckpt

# Load data (replace with appropriate path or method) sp_* = class 1; sa_* = class 0;
sp = np.load(os.path.join(data_path, 'sp_train.npy'))
sa = np.load(os.path.join(data_path, 'sa_train.npy'))
data = np.concatenate([sp, sa])
sp_lab = np.ones(len(sp))
sa_lab = np.zeros(len(sa))
labels = np.concatenate([sp_lab, sa_lab])
arr = np.arange(len(data))
np.random.seed(101)
np.random.shuffle(arr)
data_r = data[arr]
labels_r = labels[arr]

# Loading validation data
sp = np.load(os.path.join(data_path, 'sp_val.npy'))
sa = np.load(os.path.join(data_path, 'sa_val.npy'))
data = np.concatenate([sp, sa])
sp_lab = np.ones(len(sp))
sa_lab = np.zeros(len(sa))
labels = np.concatenate([sp_lab, sa_lab])
arr = np.arange(len(data))
np.random.seed(101)
np.random.shuffle(arr)
data_valid = data[arr]
labels_valid = labels[arr]

# Importing necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, concatenate, Lambda, Deconvolution2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# Define Mish Activation Function
def mish(x):
    return Lambda(lambda x: x * tf.tanh(tf.log(1 + tf.exp(x))))(x)

im_size = sp.shape[1]
data_r = np.reshape(data_r, [-1, im_size, im_size, 1])
data_valid = np.reshape(data_valid, [-1, im_size, im_size, 1])

activation1 = mish

input_size = (im_size, im_size, 1)
n = total1

# Building the initial CNN model
inputs = Input(input_size)
conv1 = Conv2D(64, 5, padding='same', activation='relu')(inputs)

for _ in range(n - 1):
    conv1 = Conv2D(128, 5, padding='same', activation='relu')(conv1)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv1)
out = Flatten()(pool2)
out = Dense(64)(out)
predictions = Dense(1, activation='sigmoid')(out)
model1 = Model(input=inputs, output=predictions)

ckpt_path = best_blackbox_ckpt 
best_ckpt = os.path.join(ckpt_path, 'best_model.hdf5')
model1.load_weights(best_ckpt)

# Adding additional layers using Deconvolution and Convolution
filter1 = 64
new_layer1 = 'no'
last_relu = 'yes'
padding1 = 'same'

max3 = model1.layers[-4].output
conv4 = Deconvolution2D(64, (2, 2), strides=2)(max3)
conv5 = Conv2D(64, 5, activation=activation1, padding='same', use_bias=False)(conv4)
conv5 = concatenate([conv5, model1.layers[-5].output])

for t1 in range(1, total1):
    conv5 = Conv2D(64, 5, activation=activation1, padding=padding1, use_bias=False)(conv5)

conv7 = Conv2D(1, 5, activation=activation1, padding='same', use_bias=False)(conv5)
flat2 = Flatten()(conv7)

# Custom ReLU function with maximum value constraint
def create_relu_advanced(max_value=1.):
    def relu_advanced(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_advanced

if last_relu == 'yes':
    dense2 = Dense(1, activation=create_relu_advanced(max_value=1.), use_bias=False)(flat2)
elif last_relu == 'no':
    dense2 = Dense(1, activation=None, use_bias=False)(flat2)

# Final model definition
model_test3 = Model(input=model1.input, output=dense2)

# Freezing layers based on 'trainable1' argument
for layer in model_test3.layers[:6]:
    layer.trainable = False
for layer in model_test3.layers[6:]:
    layer.trainable = True

# Compiling the model
model_test3.compile(optimizer=Adam(lr=1e-4, clipvalue=0.5), loss='mean_squared_error', metrics=['mean_squared_error'])

# Data preparation and augmentation
train_data_r = data_r.astype('float32')
test_data_r = data_valid.astype('float32')

train_data_aug = []
train_data_new = []

# Generating augmented data
for t1 in train_data_r:
    img = np.fliplr(t1).astype('float32')
    train_data_new.append(img)
    noise = np.random.normal(scale=0.05, size=(im_size, im_size, 1)).astype('float32')
    img = noise + t1
    img = (img - img.min()) / (img.max() - img.min())
    train_data_new.append(img.astype('float32'))

train_data_new = np.asarray(train_data_new)
train_label_r = []
test_label_r = []

# Concatenating original and augmented data
train_data_aug = np.concatenate([train_data_r, train_data_new])
train_arr = np.arange(len(train_data_aug))
np.random.seed(101)
np.random.shuffle(train_arr)
train_data_r = train_data_aug[train_arr]

# Predicting labels using the initial model
model1.load_weights(best_ckpt)
for t1 in train_data_r:
    prob = model1.predict(np.reshape(t1, [-1, im_size, im_size, 1]))
    train_label_r.append(prob)

model1.load_weights(best_ckpt)
for t1 in test_data_r:
    prob = model1.predict(np.reshape(t1, [-1, im_size, im_size, 1]))
    test_label_r.append(prob)

train_label_r = np.asarray(train_label_r)[:, 0, 0]
test_label_r = np.asarray(test_label_r)[:, 0, 0]

# Setting up callbacks for model training

# Callbacks and logging
ckpt_path = best_interpretable_ckpt 
best_ckpt = os.path.join(ckpt_path, 'best_selfinterpretable_model.hdf5')
csv_log = os.path.join(ckpt_path, 'training.log')

best_checkpoint = ModelCheckpoint(filepath=best_ckpt,
                                  monitor='val_loss',
                                  save_weights_only=True,
                                  save_best_only=True,
                                  period=1,
                                  verbose=2)
csv_logger = CSVLogger(best_ckpt + '.csv', separator=",", append=False)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

callbacks = [best_checkpoint, early_stop, csv_logger]

# Training the model
model_test3.fit(x=train_data_r, y=train_label_r, batch_size=8, epochs=50,
                validation_data=(test_data_r, test_label_r), callbacks=callbacks, shuffle=False)
