###############################################################################
# unet.py
#
# Luke Sheneman, 2023
# sheneman@uidaho.edu
#
#
# The primary tool for training a U-Net CNN model given a directory of training
# images and a corresonding directory of masks (i.e. "true" labels).
#
# This tool combines a stream of unsigned 8-bit (512x512) training
# images and their corresponding masks and trains a U-Net CNN.
#
# Most of the training parameters (Epochs, Batch size, etc.) are hardcoded.
#
# The output is a trained model and the model training history.
#
# Usage:
#  python unet.py
#
###############################################################################


import tensorflow as tf
import pickle
from math import ceil
from time import sleep, time
from random import seed
from random import random
import os
import numpy as np
from keras.models import load_model
from keras.callbacks import History

from unet_model import *   # local python file containing the U-Net architecture
from unet_data import *    # local python file containing helper functions

history = History()


print("TENSORFLOW VERSION: %s" %(tf.__version__))

# set our random seed based on current time
now = int(time())

seed(now)
np.random.seed(now)
tf.random.set_seed(now)
os.environ['PYTHONHASHSEED'] = str(now)

# Some basic training parameters
EPOCHS = 100
BATCH_SIZE = 4
TRAIN_SIZE = 300
STEPS_PER_EPOCH = ceil(TRAIN_SIZE/BATCH_SIZE)

# Create a streaming training set generator based on flows_from_directory()
traingen = trainGenerator(BATCH_SIZE,
			  'data',
			  'train_data',
			  'train_labels', 
			  flag_multi_class = True,
			  num_class = 24,
			  image_color_mode="rgb",
			  target_size=(512,512),
			  seed=now)

model = unet() # instantiate a blank U-Net CNN scaffolding

# configure a checkpoint that will be saved at the end of every epoch (if improved)
model_checkpoint = ModelCheckpoint('unet_checkpoint.h5', monitor='loss',verbose=1, save_best_only=True)

# train the U-Net CNN model 
model.fit_generator(traingen,steps_per_epoch=STEPS_PER_EPOCH,epochs=EPOCHS,callbacks=[model_checkpoint,history])

# save the trained model to disk
print("Saving model...")
model.save("./unet.model.h5")

# save the training history for the model for diagnostics
print("Saving training history...")
with open('./unet_training_history.pickle', 'wb') as histfile:
    pickle.dump(history.history, histfile)

