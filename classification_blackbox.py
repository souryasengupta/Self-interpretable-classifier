import os
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
args = parser.parse_args()
n1 = args.n

# Load data (replace with appropriate path or method) sp_* = class 1; sa_* = class 0;

sp_train = np.load('path_to_sp_train.npy')
sa_train = np.load('path_to_sa_train.npy')
data_train = np.concatenate([sp_train, sa_train])
sp_lab_train = np.ones(len(sp_train))
sa_lab_train = np.zeros(len(sa_train))
labels_train = np.concatenate([sp_lab_train, sa_lab_train])
np.random.seed(101)
np.random.shuffle(data_train)
np.random.seed(101)
np.random.shuffle(labels_train)
data_train = data_train[data_train]
labels_train = labels_train[labels_train]

sp_val = np.load('path_to_sp_val.npy')
sa_val = np.load('path_to_sa_val.npy')
data_val = np.concatenate([sp_val, sa_val])
sp_lab_val = np.ones(len(sp_val))
sa_lab_val = np.zeros(len(sa_val))
labels_val = np.concatenate([sp_lab_val, sa_lab_val])
np.random.seed(101)
np.random.shuffle(data_val)
np.random.seed(101)
np.random.shuffle(labels_val)
data_val = data_val[data_val]
labels_val = labels_val[labels_val]

# Black-box Model definition
im_size = 128
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
ckpt_path = '/path_to_save_checkpoints/'
est_ckpt = os.path.join(ckpt_path, 'model_{epoch:02d}_{val_loss:.2f}.hdf5')
best_ckpt = os.path.join(ckpt_path, 'best_model.hdf5')
csv_log = os.path.join(ckpt_path, 'training.log')

checkpoint = ModelCheckpoint(filepath=est_ckpt,
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
model.fit(data_train, labels_train,
          validation_data=(data_val, labels_val),
          callbacks=callbacks,
          epochs=50,
          batch_size=8)
