###############################################################################
# unet_model.py
#
# sheneman@uidaho.edu
#
#
# The basic Keras/TensorFlow implementation of the UNet Convolutional 
# Neural Network model as described in:
#
# Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for 
#  Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., 
#  Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention 
#  – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. 
#  Springer, Cham. 
#
#	https://doi.org/10.1007/978-3-319-24574-4_28
#
# This Keras/TensorFlow implementation adapted from:
#
#	https://github.com/zhixuhao/unet
#
################################################################################

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras


#
# Defining our U-Net architecture in Keras
#
def unet(pretrained_weights = None, input_size = (512,512,3)):

	# 512x512x3 (RGB Image) as input
	inputs = Input(input_size)

	##########################
	# THE ENCODER            #
	##########################

	# Convolutional Block #1:  Input = 512x512x3 tensor, Output = 256x256x64 tensor
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

	# Convolutional Block #2:  Input = 256x256x64 tensor, Output = 128x128x128 tensor
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	# Convolutional Block #3:  Input = 128x128x128 tensor, Output = 64x64x256 tensor
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	# Convolutional Block #4:  Input = 64x64x256 tensor, Output = 32x32x512 tensor
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)   # help prevent model overfitting
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)




	##########################
	# THE BOTTLENECK         #
	##########################
	
	# Convolutional block #5:  Input = 32x32x512 tensor, Output = 32x32x1024 tensor
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)  # help prevent model overfitting




	##########################
	# THE DECODER            #
	##########################

	# Convolutional block #6:  Input = 32x32x1024 tensor, Output = 64x64x512 tensor
	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)	# skip connection -
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	# Convolutional block #7:  Input = 64x64x512 tensor, Output = 128x128x246 tensor
	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)	# skip connection!!
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	# Convolutional block #8:  Input = 128x128x256 tensor, Output = 256x256x128 tensor
	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)	# skip connection!!
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	# Convolutional block #9:  Input = 256x256x128, Output = 512x512x64 tensor
	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)	# skip connection
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)


	####################################################################################
	# OUTPUT LAYER   								   #
	# one filter for each of the classes (e,g, 24 in this case)                        #
	#	                                                                           #
	# Convolutional block #10:  Input = 512x512x64 tensor, Output = 512x512x24 tensor  #
	####################################################################################
	conv10 = Conv2D(24, 1, padding='same', activation='softmax')(conv9)

	model = Model(inputs, conv10)

	# multi-class segmentation problem, so use a categorical_crossentropy loss function	
	model.compile(optimizer =  Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = [tf.keras.metrics.AUC()])
   
	# print a readable verion of the model architecture 
	model.summary()

	# if pre-trained weights are provided, load those into our model architecture
	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

