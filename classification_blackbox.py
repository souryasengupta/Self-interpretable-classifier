
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1,
                    help="Number of convolutional layers")
parser.add_argument("--data_path", type=str, default="/path/to/your/data_directory",
                    help="Path to the directory containing training and validation data")
parser.add_argument("--best_blackbox_ckpt", type=str, default="/path/to/your/data_directory",
                    help="Path to the best checkpoint file")
args = parser.parse_args()
n1 = args.n
data_path = args.data_path
best_blackbox_ckpt = args.best_blackbox_ckpt


# Load data (replace with appropriate path or method) sp_* = class 1; sa_* = class 0;

sp_train_path = os.path.join(args.data_path, 'sp_train.npy')
sa_train_path = os.path.join(args.data_path, 'sa_train.npy')
sp_val_path = os.path.join(args.data_path, 'sp_val.npy')
sa_val_path = os.path.join(args.data_path, 'sa_val.npy')


sp = np.load(sp_train_path)
sa = np.load(sa_train_path)
data = np.concatenate([sp, sa])
sp_lab = np.ones(len(sp))
sa_lab = np.zeros(len(sa))
labels = np.concatenate([sp_lab,sa_lab])
arr = np.arange(len(data))
np.random.seed(101)
np.random.shuffle(arr)
data_r = data[arr]
labels_r = labels[arr]

sp = np.load(sp_val_path)
sa = np.load(sa_val_path)
data = np.concatenate([sp, sa])
sp_lab = np.ones(len(sp))
sa_lab = np.zeros(len(sa))
labels = np.concatenate([sp_lab,sa_lab])
arr = np.arange(len(data))
np.random.seed(101)
np.random.shuffle(arr)
data_valid = data[arr]
labels_valid = labels[arr]

im_size = sp.shape[1]
data_r = np.reshape(data_r, [-1, im_size, im_size, 1])
data_valid = np.reshape(data_valid, [-1, im_size, im_size, 1])

# Black-box Model definition

input_size = (im_size, im_size, 1)
n = int(n1)
inputs = Input(input_size)
conv1 = Conv2D(64, 5, padding='same', activation='relu')(inputs)
for _ in range(n-1):
    conv1 = Conv2D(128, 5, padding='same', activation='relu')(conv1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv1)
out = Flatten()(pool2)
out = Dense(64)(out)
predictions = Dense(1, activation='sigmoid')(out)
model = Model(inputs=inputs, outputs=predictions)
model.summary()

# Model compilation
model.compile(optimizer=Adam(lr=3e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks and logging
ckpt_path = best_blackbox_ckpt 
best_ckpt = os.path.join(ckpt_path, 'best_model.hdf5')
csv_log = os.path.join(ckpt_path, 'training.log')

checkpoint = ModelCheckpoint(filepath=best_ckpt,
                             monitor='val_loss',
                             save_weights_only=True,
                             save_best_only=False,
                             verbose=1)
best_checkpoint = ModelCheckpoint(filepath=best_ckpt,
                                  monitor='val_loss',
                                  save_weights_only=True,
                                  save_best_only=True,
                                  verbose=1)
csv_logger = CSVLogger(csv_log, separator=',', append=False)

early_stop = EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           mode='min',
                           restore_best_weights=True)

callbacks = [checkpoint, best_checkpoint, early_stop, csv_logger]

# Model training
model.fit(data_r, labels_r,
          validation_data=(data_valid, labels_valid),
          callbacks=callbacks,
          epochs=50,
          batch_size=8)
