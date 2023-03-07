##############################################################################
# unet_data.py
#
# Luke Sheneman, 2023
# sheneman@uidaho.edu
#
# Support and wrapper functions for streaming training and test data, loading
# multi-class labels, and saving classification output.
#
################################################################################

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os, sys
import skimage.io as io
from skimage import img_as_ubyte
import tensorflow as tf

# for debugging
np.set_printoptions(threshold=sys.maxsize)


###############################################################################
# 
# trainGenerator()
#
# A wrapper function for constructing a Keras/TensorFlow training set by 
# streaming both the raw and mask (i.e. label) data from directories, 
# zipping them together for use in the UNet CNN model.
#
###############################################################################
def trainGenerator(batch_size, train_path, image_folder, mask_folder,
		   image_color_mode = "grayscale", mask_color_mode = "grayscale",
		   image_save_prefix  = "image", mask_save_prefix  = "mask",
		   flag_multi_class = False, num_class = 2, target_size = (512,512), seed = 1):

	image_datagen = ImageDataGenerator()
	image_generator = image_datagen.flow_from_directory(
		train_path,
		classes = [image_folder],
		class_mode = None,
		color_mode = image_color_mode,
		target_size = target_size,
		batch_size = batch_size,
		save_prefix  = image_save_prefix,
		seed = seed)

	mask_datagen = ImageDataGenerator()
	mask_generator = mask_datagen.flow_from_directory(
		train_path,
		classes = [mask_folder],
		class_mode = None,
		color_mode = mask_color_mode,
		target_size = target_size,
		batch_size = batch_size,
		save_prefix  = mask_save_prefix,
		seed = seed)

	# combine flows using zip
	train_generator = zip(image_generator, mask_generator)

	# yield the image and one-hot encoded mask (which is essential for multi-class data)
	for (img, mask) in train_generator:
		img  = img/255
		mask = tf.one_hot(mask, 24, dtype=tf.int32)
		mask = mask[:, :, :, 0, :]
		yield(img, mask)




###############################################################################
#
# testGenerator()
#
# A function for streaming test images during the classification step.
#
###############################################################################
def testGenerator(testdir_path, filenames):
	for f in filenames:
		fullpath = testdir_path + '/' + f
		img = io.imread(fullpath,as_gray = False)
		img = img / 255
		img = np.reshape(img,(1,)+img.shape)

		yield img



###############################################################################
#
# loadMasks()
#
# Load just the masks (i.e. "true" binary labels) separately
#
###############################################################################
def loadMasks(maskdir_path, filenames):
	mask_vector = []
	for f in filenames:
		fullpath = os.path.join(maskdir_path, f)
		img = io.imread(fullpath) / 255; 
		mask_vector = np.append(mask_vector, img.flatten())

	return(mask_vector)




###############################################################################
#       
# saveResults()
#               
# Save inference image to disk in unsigned 8-bit form.
#               
###############################################################################
def saveResult(save_path,filenames,results):
	for i,item in enumerate(results):
		p = np.argmax(item, axis=-1)  	# collapse the one-hot encoding to grayscale image
		io.imsave(os.path.join(save_path,filenames[i]),img_as_ubyte(p))  # save as 8-bit image
