##############################################################################
#
# convert.py
#
# Luke Sheneman, 2023
# sheneman@uidaho.edu
#
# Convert raw grayscale output from unet_inference.py to RGB encoded version
# based on class dictionary
#
##############################################################################

import os, sys
import cv2
import numpy as np
import csv

HEIGHT = 512
WIDTH  = 512

NUM_CLASSES = 24

INPUT_DIR  		= "./raw_output"
OUTPUT_DIR 		= "./final_output"
CLASS_DICTIONARY_FILE 	= "./data/class_dict_seg.csv"

################################################################
#
# open our class dictionary CSV and read it. 
#
################################################################
csvfile = open(CLASS_DICTIONARY_FILE, "r")
next(csvfile)   # skip header
class_dict = list(csv.reader(csvfile, delimiter=","))
csvfile.close()

################################################################
#
# for every file in the INPUT_DIR directory, use the 
# provide class dictonary to map the grayscale class number
# to an RGB tuple for appropriately color-coded images
#
################################################################
files = os.listdir(INPUT_DIR)
for f in files:
	fp = os.path.join(INPUT_DIR, f)
	img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

	blank_image = np.zeros((HEIGHT,WIDTH,3), np.uint8)
	for i in range(HEIGHT):
		for j in range(WIDTH):
			index = img[i,j]
			if(index < NUM_CLASSES):
				(name,r,g,b) = class_dict[index]
				blank_image[i,j] = (r,g,b)

	final_path = os.path.join(OUTPUT_DIR, f)
	cv2.imwrite(final_path, blank_image)
	print(f)


