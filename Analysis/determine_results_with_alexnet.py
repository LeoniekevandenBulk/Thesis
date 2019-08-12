# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:15:33 2019

@author: Leonieke

Script to collect the MSE between the Alexnet features of the true test video and
a reconstruction of that video.
"""

import os
import cv2
import torch
import pickle
import numpy as np
from alexnet_model import alexnet
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import euclidean

# Set locations of the videos
test_video_path = "GroundTruth_TestVideo_Delay8.mp4"
reconstruction_video_path = "ME_Reconstructions_Top100_with_normalization/Reconstruction_TestVideo_Top100_Concatenated.mp4" # Change this for different reconstructions

# Set save names
reconstruction_results_name = "ME_Alexnet_correlation_analysis.pkl"

# Load the videos with cv2 
test_capture = cv2.VideoCapture(os.path.join(os.getcwd(),test_video_path))
reconstruction_capture = cv2.VideoCapture(os.path.join(os.getcwd(),reconstruction_video_path))

# Check if number of frames is equal across videos
test_count = int(test_capture.get(cv2.CAP_PROP_FRAME_COUNT))
reconstruction_count = int(reconstruction_capture.get(cv2.CAP_PROP_FRAME_COUNT))
if(not(test_count == reconstruction_count)):
    raise RuntimeError("Number of frames across videos is not equal")

# Initialize model and move to GPU
model = alexnet(pretrained=True, progress=True, return_activations=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
model.to(device)
model.eval()

# Go through the reconstrution and test video frame by frame, feed them to the
# model and save the similarity between the features on each layer between the 
# reconstruction and the test video
reconstruction_results = {"Layer1": [], "Layer2": [], "Layer3": [], "Layer4": [],
                          "Layer5": [], "Layer6": [], "Layer7": [], "Layer8": []}
for i in range(test_count):
    # Print progress
    print(i)
    
    # Read a single frame from each video
    test_success, test_frame = test_capture.read()
    reconstruction_success, reconstruction_frame = reconstruction_capture.read()
    if(not(test_success and reconstruction_success)):
       raise RuntimeError("Frame could not be read")
       
    # Resize the frames to the appropriate size and switch from BGR to RGB
    test_frame = cv2.resize(test_frame, (224, 224))
    test_frame = test_frame[:,:,(2,1,0)]
    test_frame = test_frame.astype(np.float32)
    reconstruction_frame = cv2.resize(reconstruction_frame, (224, 224))
    reconstruction_frame = reconstruction_frame[:,:,(2,1,0)]
    reconstruction_frame = reconstruction_frame.astype(np.float32)
    
    # Change the frame to a range of [0,1] and normalize with mean and std
    test_frame = (test_frame-np.min(test_frame))/(np.max(test_frame)-np.min(test_frame))
    test_frame = test_frame - np.array([[[0.485, 0.456, 0.406]]]) # Subtract mean
    test_frame = test_frame / np.array([[[0.229, 0.224, 0.225]]]) # Divide by std
    reconstruction_frame = (reconstruction_frame-np.min(reconstruction_frame))/(np.max(reconstruction_frame)-np.min(reconstruction_frame))
    reconstruction_frame = reconstruction_frame - np.array([[[0.485, 0.456, 0.406]]]) # Subtract mean
    reconstruction_frame = reconstruction_frame / np.array([[[0.229, 0.224, 0.225]]]) # Divide by std
        
    # Tranpose matrix and transform to right shape (batch_size=1 x 3 x height x width)
    test_frame = test_frame.transpose((2, 0, 1))
    test_frame = np.expand_dims(test_frame, axis=0)
    reconstruction_frame = reconstruction_frame.transpose((2, 0, 1))
    reconstruction_frame = np.expand_dims(reconstruction_frame, axis=0)

    # Tranform frames to pytorch Tensor
    test_frame_input = torch.from_numpy(test_frame)
    reconstruction_frame_input = torch.from_numpy(reconstruction_frame)
    
    # Feed input to device and model
    test_frame_input = test_frame_input.to(device, dtype=torch.float)
    with torch.no_grad():
        test_activations = model(test_frame_input)
    reconstruction_frame_input = reconstruction_frame_input.to(device, dtype=torch.float)
    with torch.no_grad():
        reconstruction_activations = model(reconstruction_frame_input)
        
    # Go through all layers and compare similarity by checking the normalized, squared Euclidian distance (MSE)
    for num,layer in enumerate(reconstruction_results):
        test_activation = np.ravel(test_activations[num].cpu().detach().numpy())
        reconstruction_activation = np.ravel(reconstruction_activations[num].cpu().detach().numpy())
        similarity = pearsonr(test_activation,reconstruction_activation)[0]
        reconstruction_results[layer].append(similarity)
        
# Release cv2 captures
test_capture.release()
reconstruction_capture.release()

# Print average of the errors per layer
print("This reconstruction had the following average correlation per layer:")
for layer in reconstruction_results:
    avg = np.mean(reconstruction_results[layer])
    print(layer + " had a average correlation of: " + str(avg))

#Save list as pickle to disk
reconstruction_save_path = os.path.join(os.getcwd(),reconstruction_results_name)
reconstruction_pkl_file = open(reconstruction_save_path,'wb')
pickle.dump(reconstruction_results,reconstruction_pkl_file,protocol=2)
reconstruction_pkl_file.close()  