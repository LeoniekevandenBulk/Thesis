# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:40:32 2019

@author: Leonieke

Script to determine the top 100 best prior videos from a parallel set of prior videos
for each TR in the test video for all three models (ME only, NN_pool only and combined).
The NN_pool prior features are made on the fly.  The top 100 videos are based on the 
correlation between the predicted fmri response of the prior video, on basis of 
either their Motion Energy features or their NN features,and the actual fmri 
response of that TR of the test video. Which feature domain is chosen in the 
combined model is based on if a voxel was better correlated with the ME features 
or the NN features.
"""

import os
import sys
import cv2
import time
import torch
import pickle
import scipy.io
import hdf5storage
import numpy as np
from scipy.stats.stats import pearsonr
from Adapted_R2Plus1D_model import R2Plus1DClassifier

# Add a parameter containing the name of a txt file containing the part of the 
# prior files to be processed in this run (based on files in MotionEnergyPriorClips)
txt_file = sys.argv[1] # e.g. "../Prior/ME_prior_names0.txt"

# Set voxel selection file (output from select_best_delay.py)
ME_encoding_model_file = "ME_voxel_selection_alldelays_2000.mat"
NN_encoding_model_file = "NN_Pool_voxel_selection_alldelays_2000.mat"
combined_encoding_model_file = "combined_voxel_selection_alldelays_2700.mat"

# Set what to resize the video size to for the NN features
size = (112,112)

# Set location of the directory of the prior files to be used
ME_prior_dir = "../Motion-energy/MotionEnergyPriorClips"
NN_prior_dir = "../Prior/Clips" #"Clips"

# Set a directory to save the dictionaries to be created for the top 100 prior 
# files and a name to save the dictionaries as 
ME_save_dir = "ME_Top_100_Prior_Dictionaries_testset"
ME_save_name = "ME_top_100_best_clips" + txt_file.split(".")[0][-1] + ".pkl"
ME_save_path = os.path.join(os.getcwd(),ME_save_dir,ME_save_name)
NN_save_dir = "NN_Top_100_Prior_Dictionaries_testset" 
NN_save_name = "NN_top_100_best_clips" + txt_file.split(".")[0][-1] + ".pkl"
NN_save_path = os.path.join(os.getcwd(),NN_save_dir,NN_save_name)
combined_save_dir = "Combined2700_Top_100_Prior_Dictionaries_testset"
combined_save_name = "Combined2700_top_100_best_clips" + txt_file.split(".")[0][-1] + ".pkl"
combined_save_path = os.path.join(os.getcwd(),combined_save_dir,combined_save_name)

# Check if save directories already exist, else make it
if not(os.path.isdir(os.path.join(os.getcwd(), ME_save_dir))):
    os.makedirs(os.path.join(os.getcwd(), ME_save_dir))
if not(os.path.isdir(os.path.join(os.getcwd(), NN_save_dir))):
    os.makedirs(os.path.join(os.getcwd(), NN_save_dir))
if not(os.path.isdir(os.path.join(os.getcwd(), combined_save_dir))):
    os.makedirs(os.path.join(os.getcwd(), combined_save_dir))

# Make a array of the paths to the files
txt_path = os.path.join(os.getcwd(),txt_file)
ME_file_list = []
NN_file_list = []
with open(txt_path, 'r') as txt:
    for line in txt:
        ME_file_list.append(os.path.join(os.getcwd(),ME_prior_dir,line.strip('\n')))
        NN_file = line.split("_")[-1].split(".")[0] + ".mp4"
        NN_file_list.append(os.path.join(os.getcwd(),NN_prior_dir,NN_file))

# Load encoding model which only contains the chosen voxel selection
print("Loading ME encoding model")
ME_encoding_model = hdf5storage.loadmat(ME_encoding_model_file)
ME_weights = ME_encoding_model['W']
ME_test_fmri = ME_encoding_model['test_fmri']

print("Loading NN encoding model")
NN_encoding_model = hdf5storage.loadmat(NN_encoding_model_file)
NN_weights = NN_encoding_model['W']
NN_test_fmri = NN_encoding_model['test_fmri']

print("Loading Combined encoding model")
combined_encoding_model = hdf5storage.loadmat(combined_encoding_model_file)
combined_ME_weights = combined_encoding_model['ME_W']
combined_NN_weights = combined_encoding_model['NN_W']
combined_model_per_voxel = combined_encoding_model['model']
combined_test_fmri = combined_encoding_model['test_fmri']

# Make model and transfer to GPU/CPU
model = R2Plus1DClassifier(400, (2, 2, 2, 2), pretrained=False, finetuned=True, return_activations=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
model.to(device)
model.eval()

# We want to save the top 100 best fitting clips per TR over all test videos based 
# on the correlation between the measured and predicted fMRI. Here we initialize
# the top 100 matrices.
ME_top_clips = {}
NN_top_clips = {}
combined_top_clips = {}
for TR in range(ME_test_fmri.shape[0]):
    ME_top = []
    NN_top = []
    combined_top = []
    for i in range(100):
        ME_top.append((-1,''))
        NN_top.append((-1,''))
        combined_top.append((-1,''))
    ME_top_clips[str(TR)] = ME_top
    NN_top_clips[str(TR)] = NN_top
    combined_top_clips[str(TR)] = combined_top
    
# Go over each file, transform video to numpy array, calculate R2Plus1D features and save to disk
start_time = time.time() # Time to see how long getting the features takes
for num in range(len(ME_file_list)):
    
    # Print progress
    if(num%100 == 0):
        print(num)
    
    # Save file name
    ME_file_path = ME_file_list[num]
    ME_file_name = ME_file_path.split('/')[-1]
    NN_file_path = NN_file_list[num]
    NN_file_name = NN_file_path.split('/')[-1]
    print(NN_file_name)
    
    # Get features from NN on the fly
    try:
        # Transform video to numpy array
        capture = cv2.VideoCapture(NN_file_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Frames: " + str(frame_count) + ", Width: " + str(frame_width) + ", Height: " + str(frame_height))
        
        if(not (frame_width == size[0] and frame_height == size[1])):
            continue
        
        frames = np.empty((16, frame_height, frame_width, 3), np.dtype('float32'))
        for ind in range(16):
            ret, frame = capture.read()
            frame = frame[:,:,(2,1,0)] # Swap from BGR to RGB encoding
            frames[ind] = frame.astype(np.float32)
        capture.release()
   
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
    
    except:
        print(NN_file_name + " could not be read")
        pass
    
    # Get the feature to CPU and in numpy form
    NN_feature = activations[5].cpu().detach().numpy() # In this case 5 is the pooling layer 
    
    # Only get first timepoint from feature to have a sampling rate of 1 HZ
    NN_feature = NN_feature[0][:,0,:,:]
    
    # Reshape feature to a flat array
    NN_feature = np.squeeze(NN_feature)
    NN_feature = np.ravel(NN_feature)
    
    # Load the ME feature to an array
    try:
        ME_feature = scipy.io.loadmat(ME_file_path)['S_fin'] #hdf5storage.loadmat(ME_file_path)['S_fin']
    except:
        print(ME_file_name + " could not be read")
        continue
    
    # Predict fMRI response belonging to this feature by multiplying with the 
    # correspondig weights per voxel for both the ME and NN voxels
    ME_prediction = []
    NN_prediction = []
    combined_prediction = []
    for w_num in range(ME_weights.shape[1]):
        pred = np.matmul(ME_feature,ME_weights[:,w_num])[0]
        ME_prediction.append(pred)
        
    for w_num in range(NN_weights.shape[1]):
        pred = np.matmul(NN_feature,NN_weights[:,w_num])
        NN_prediction.append(pred)
        
    for w_num in range(combined_ME_weights.shape[1]):
        if(combined_model_per_voxel[w_num] == 0):
            pred = np.matmul(ME_feature,combined_ME_weights[:,w_num])[0]
        else:
            pred = np.matmul(NN_feature,combined_NN_weights[:,w_num])
        combined_prediction.append(pred)
    
    # Check correlation between predicted and true fMRI response and check if 
    # it belongs in the top 100 best clips per TR per test video
    for TR in ME_top_clips:
        # Set true responses
        ME_true_response = ME_test_fmri[int(TR),:] 
        NN_true_response = NN_test_fmri[int(TR),:]
        combined_true_response = combined_test_fmri[int(TR),:]
        
        # Check correlation
        ME_corr = pearsonr(ME_prediction,ME_true_response)[0]
        NN_corr = pearsonr(NN_prediction,NN_true_response)[0]
        combined_corr = pearsonr(combined_prediction,combined_true_response)[0]
        
        # Check if correlation is higher than the current lowest of the top 100,
        # replace that one and sort again
        if(ME_corr > ME_top_clips[TR][99][0]):
            ME_top_clips[TR][99] = (ME_corr,NN_file_name.split('.')[0])
            ME_top_clips[TR] = sorted(ME_top_clips[TR],reverse=True)
        if(NN_corr > NN_top_clips[TR][99][0]):
            NN_top_clips[TR][99] = (NN_corr,NN_file_name.split('.')[0])
            NN_top_clips[TR] = sorted(NN_top_clips[TR],reverse=True)
        if(combined_corr > combined_top_clips[TR][99][0]):
            combined_top_clips[TR][99] = (combined_corr,NN_file_name.split('.')[0])
            combined_top_clips[TR] = sorted(combined_top_clips[TR],reverse=True)

    # Save the top 100 every 10000 prior features in case the program needs to be stopped
    if(num%10000 == 0):
        ME_pkl_file = open(ME_save_path,'wb')
        pickle.dump(ME_top_clips,ME_pkl_file,protocol=2)
        ME_pkl_file.close()  
        
        NN_pkl_file = open(NN_save_path,'wb')
        pickle.dump(NN_top_clips,NN_pkl_file,protocol=2)
        NN_pkl_file.close()  
        
        combined_pkl_file = open(combined_save_path,'wb')
        pickle.dump(combined_top_clips,combined_pkl_file,protocol=2)
        combined_pkl_file.close()
        
# Print the time
duration = time.time()-start_time  
print("This took " + str(duration) + " second. This is " + str(duration/3600) + " hours.")

#Save the dictionary containing the top 100 clips per TR per test video to disk
ME_pkl_file = open(ME_save_path,'wb')
pickle.dump(ME_top_clips,ME_pkl_file,protocol=2)
ME_pkl_file.close()  

NN_pkl_file = open(NN_save_path,'wb')
pickle.dump(NN_top_clips,NN_pkl_file,protocol=2)
NN_pkl_file.close()  

combined_pkl_file = open(combined_save_path,'wb')
pickle.dump(combined_top_clips,combined_pkl_file,protocol=2)
combined_pkl_file.close()   