# Semantic Segmentation of Aerial Drone Imagery in TensorFlow/Keras
Some code for the Semantic Segmentation in TensorFlow/Keras Data Hub TechTalk, March 7 2023

# Installation Instructions
1. clone this repo
2. download the data (see below) and structure like this:

  data  
  ├── class_dict_seg.csv  
  ├── RGB_color_image_masks  
  ├── test_data  
  ├── test_labels  
  ├── train_data  
  ├── train_labels  
  ├── val_data  
  └── val_labels  
  
The program assumes all images are *.png files that have been resized and cropped to 512x512

3. Create virtual environment (python3 -m venv venv)
4. source ./venv/bin/activate
5. pip install -U pip
6. pip install tensorflow opencv-python scikit-image
7. mkdir raw_output final_output

# Training
1. source ./venv/bin/activate
2. python unet.py

# Inference
1. source ./venv/bin/activate
2. python ./unet_inference.py  (inferred output will be in "raw_output")

# Create RGB-encoded visible results from inferences
1. source ./venv/bin/activate
2. python ./convert.py   (files will be in "final_output")

## Data Availability
https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

More information on the dataset can be found here:
http://dronedataset.icg.tugraz.at/

