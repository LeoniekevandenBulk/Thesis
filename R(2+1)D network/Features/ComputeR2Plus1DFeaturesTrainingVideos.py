# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:35:34 2019

@author: Leonieke

Script to compute the features from the R2Plus1D neural network from the 
training videos.  The directory of the training videos and the directory of
where to write the features to can be specified in the 'from_dir' and 'to_dir' 
variables below.
"""

import os
import cv2
import sys
import errno
import torch
import hdf5storage
import numpy as np
from Adapted_R2Plus1D_model import R2Plus1DClassifier

# Set directories to read from and write features to
from_dir = "../Data/DoctorWho_TrainingVideos"
to_dir = "R2Plus1DTrainingVideos"
save_name_base = "R2Plus1D_TrainingVideo"

# Set layer names
layer_names = ["Conv1","Conv2","Conv3","Conv4","Conv5","Pool"]

# Set what to resize the video size to
resize = (112,112)

# Check save path to save clips to
if not(os.path.isdir(to_dir)):
    os.makedirs(to_dir)

# Make list of locations of files
cur_dir = os.getcwd()
file_list = os.listdir(os.path.join(cur_dir,from_dir))

# Make model and transfer to GPU/CPU
model = R2Plus1DClassifier(400, (2, 2, 2, 2), pretrained=False, finetuned=True, return_activations=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Device being used:", device)
model.to(device)
model.eval()

# Go over each file, transform video to numpy array, calculate R2Plus1D features and save to disk
for file_name in file_list:
    
    # Print file name and complete path
    print(file_name)
    file_path = os.path.join(cur_dir,from_dir,file_name)
    
    try:
        print(file_path)
        # Transform video to numpy array
        capture = cv2.VideoCapture(file_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Frames: " + str(frame_count) + ", Width: " + str(frame_width) + ", Height: " + str(frame_height))
        
        # Calculate amount of individual reponses that need to be extracted from the training video
        nr_responses = int(np.floor(frame_count/16))
        print(nr_responses)
        
        responses = {}
        for name in layer_names:
            responses[name] = []
        
        # Loop over the video and take 16 frames * nr_responses times
        for nr in range(nr_responses):
            
            # Save 16 frames in numpy array
            frames = np.empty((16, resize[0], resize[1], 3), np.dtype('float32'))
            for ind in range(16):
                ret, frame = capture.read()
                frame = cv2.resize(frame, (resize[0], resize[1]))
                frame = frame[:,:,(2,1,0)] # Swap from BGR to RGB encoding
                frames[ind] = frame.astype(np.float32)
            
            # Normalization
            for i, frame in enumerate(frames):
                frame = frame - np.array([[[110.201, 100.64, 95.9966]]]) # Subtract mean
                frame = frame / np.array([[[58.1489, 56.4701, 55.3324]]]) # Divide by std
                frames[i] = frame
                
            # Tranpose matrix and transform to right shape (batch_size x 3 x frames x height x width)
            frames = frames.transpose((3, 0, 1, 2))
            frames = np.expand_dims(frames, axis=0)
        
            # Tranform frames to pytorch Tensor
            frames_input = torch.from_numpy(frames)
            
            # Feed input to device and model
            frames_input = frames_input.to(device)
            with torch.no_grad():
                activations = model(frames_input)

            # Append layer activations to reponse dictionary
            for nr,layer in enumerate(layer_names):
                responses[layer].append(activations[nr].cpu().detach().numpy())
        
        # Release the video capture
        capture.release()
        
        # Save activations to disk
        print("Saving activations to disk")
        save_name = save_name_base + file_name.split('.')[0] + ".mat"
        features = {}
        for layer in layer_names:
            
            # First get shape of features and make empty matrix
            feature_shape = responses[layer][0][0].shape
            layer_features = np.zeros((nr_responses,feature_shape[0]*feature_shape[2]*feature_shape[3]))
            print(layer_features.shape)
            
            # Loop over activations within a layer type
            for nr in range(nr_responses):

                #  Save the first frame of the feature maps
                feature = responses[layer][nr][0][:,0,:,:]
                
                # Squeeze and reshape to 2D vector
                feature = np.squeeze(feature)
                feature = np.ravel(feature)
                #feature = feature.reshape((feature_shape[0]*feature_shape[2]*feature_shape[3]))

                # Save to matrix
                layer_features[nr,:] = feature
                
            features[layer] = layer_features
                
        # Save the feature matrix to the 'to_dir' directory
        hdf5storage.savemat(os.path.join(cur_dir,to_dir,save_name), features)
        
    
    except KeyboardInterrupt:
        sys.exit() # Exit when ctrl+c is pressed
        
    except OSError as error:
        if error.errno == errno.ENOSPC:
            print("Disk is full!")
            sys.exit() # Exit if disk is full or file can't be written
        else:
            print("OS Error")
            pass
        
    except Exception as e:
        print(file_name + " could not be read")
        print(e)
        pass