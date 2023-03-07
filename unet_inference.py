###############################################################################
# unet_inference.py
#
# Luke Sheneman, 2023
# sheneman@uidaho.edu
#
# This tool will load a trained and saved Keras/TensorFlow model from disk and 
# stream a set of raw unsigned 8-bit, pre-padded (256x256), images through the 
# UNET classifier in order to generate a set of binary semantic segmentation 
# maps for each image.  It will save the resulting binary output images to disk 
# in the specified output folder.   Paths and filenames are hardcoded below 
# (for now).
#
# Initial 32-bit float raw images must be initially preprocessed:
#    1) Scaled to normalized 8-bit unsigned representation
#    2) Padded to be 256x256 (see pad.py)
#
# Usage:
# python unet_classify.py
#
################################################################################

from unet_model import *   # local python file containing the UNet architecture
from unet_data import *    # local python file containing helper functions
from time import time
import os
import tensorflow as tf

# The base directory path for reading/writing files
BASEDIR = "."

print("TENSORFLOW VERSION: %s" %(tf.__version__))


#
# Some custom callbacks to print diagnotic telemetry 
#
class CustomCallback(tf.keras.callbacks.Callback):
	def on_predict_batch_begin(self, batch, logs=None):
		global start_time
		start_time = time()

	def on_predict_batch_end(self, batch, logs=None):
		global start_time
		end_time = time()
		print(end_time - start_time)


model  = load_model("unet_checkpoint.h5")

testdir_path = os.path.join("data/test_data") 
filenames = os.listdir(testdir_path)

start_time = 0

# stream files through the classifier using an ImageDataGenerator.flow_from_directory()
testGen = testGenerator(testdir_path, filenames)

# perform classification using our streaming images from our test generator
results = model.predict_generator(testGen,steps=len(filenames),callbacks=[CustomCallback()],verbose=1)

outdir_path = os.path.join(BASEDIR,"raw_output")

# Save our results to disk
saveResult(outdir_path,filenames,results)

