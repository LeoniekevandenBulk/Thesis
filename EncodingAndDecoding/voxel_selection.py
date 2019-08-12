# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:14:12 2019

@author: Leonieke

Script to save the top X voxels and their optimal hemodynamic delay from all collected 
voxels based on picking the highest correlation from the encoding models on a validation set.
With the top X voxels, the learned weights from the encoding model and the corresponing test
fmri signals are saved in a dictionary.
"""

# Import packages
import os
import cv2
import hdf5storage
import numpy as np

# Set a number for the top X voxels and set the delays to test
X = 2000
delays = [4,5,6,7,8]

# Set the encoding model files and the name to save the voxel selection to
encoding_file_name = "lbulk_full_ME_delay4__corrs_whodata_from_1_to_121.mat" # if more than one delay, name the first of the file names
if(len(delays)==1):
    save_name = "ME_voxel_selection_delay" + str(delays[-1]) + "_" + str(X) + ".mat"
else:
    save_name = "ME_voxel_selection_alldelays_" + str(X) + ".mat"

# Set the directory to the test videos and whether to make the corresponding ground truth for the selected delays
test_dir = "../Data/DoctorWho_TestVideos"
make_test_video = True

# Print info
print("Selecting the top " + str(X) + " voxels within delays of " + str(delays))

# First process the test videos to frames and delete frames that won't be reconstructed
# by the highest delay, as they all need to be processed for the same TRs
test_video_list = os.listdir(os.path.join(os.getcwd(),test_dir))
test_videos = {}
video_TR_lengths = {}
for video_file in test_video_list:
    
    # Set path and load video with opencv
    video_path = os.path.join(os.getcwd(),test_dir,video_file)
    capture = cv2.VideoCapture(video_path)
    
    # Calculate how many TR series are in the test video and make dictionary to save frames in
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    nr_TRs = int(np.floor(frame_count/16))
    video_TR_lengths[video_file] = nr_TRs
    
    # Only save frames of test videos if the ground truth test video needs to be made
    if(make_test_video):
    
        # Loop over TRs
        TRs = {}
        for nr in range(nr_TRs):
            
            # Save 16 frames in numpy array as that corresponds to one TR
            frames = np.empty((16, 112, 112, 3), np.dtype('float32'))
            for ind in range(16):
                ret, frame = capture.read()
                frame = cv2.resize(frame, (112, 112))
                frames[ind] = frame.astype(np.float32)
            TRs[nr] = frames
        
        # Delete TRs based on longest delay (the longest delay amount of frames removed from the back)
        for nr in range(2*delays[-1]):
            TRs.pop(nr_TRs-(nr+1))
        
        # Save video frames to dict
        test_videos[video_file] = TRs
        
    # Release the video capture
    capture.release()
 
# If ground truth test video needs to be made, concatenate all video frames from the different test videos and save them as one video
if(make_test_video):
    video_save_name = "GroundTruth_TestVideo_Delay" + str(delays[-1]) + ".mp4"
    save_path = os.path.join(os.getcwd(), video_save_name)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m','p','4','v'), 22.86, (112,112))
    
    # Loop over all frames and write them
    for video_file in test_video_list:
        for TR in test_videos[video_file]:
            frames = test_videos[video_file][TR]
            for nr in range(16):
                frame = frames[nr]
                out.write(np.uint8(frame))
    out.release()

# Set dicionaries to save top correlations, weights and test fmri signals per delay
delay_dict = {}

# Loop over all delays and collect the top X correlations per delay and save those with corresponding weights and test fmri signals
for delay in delays:
    print(delay)
    
    # Load encoding model file for right delay
    file_name = encoding_file_name.replace("delay"+str(delays[0]),"delay"+str(delay))
    delay_file = hdf5storage.loadmat(os.path.join(os.getcwd(),file_name))
    
    # Save correlations, weights and test fmri signals
    corrs = delay_file['corrs_val']
    W = delay_file['W']
    test_fmri = delay_file['realX_test']
    
    # Sort correlation zipped with their positions and take the top X
    pos = range(corrs.shape[0])
    d = [delay] * corrs.shape[0]
    sort = sorted(zip(corrs,pos,d),reverse=True)
    top = sort[0:X]
    
    # Save weights and test fmri signals corresponding to the top X positions
    top_W = np.zeros((W.shape[0],X)) 
    top_test_fmri = np.zeros((test_fmri.shape[0],X))
    for num,tup in enumerate(top):
        top_W[:,num] = W[:,tup[1]]
        top_test_fmri[:,num] = test_fmri[:,tup[1]]
    
    # Save all data in the dictionary on position 'delay'
    delay_dict[delay] = {}
    delay_dict[delay]['corrs_and_pos'] = top
    delay_dict[delay]['W'] = top_W
    delay_dict[delay]['test_fmri'] = top_test_fmri

# Append all the correlations and positions from all delays and sort by correlation
top = []
array_pos = []
for delay in delays:
    top = top + delay_dict[delay]['corrs_and_pos']
    array_pos.extend(list(range(X)))
top = sorted(zip(top,array_pos),reverse=True)

# Create matrices to save the best weights, test fmri and positions in based on the best correlations over all delays
top_W = np.zeros((W.shape[0],X)) # Initialize matrix to save the final weights and positions in
top_test_fmri = np.zeros((test_fmri.shape[0],X))
top_pos = np.zeros((X))
top_corr = np.zeros((X))
top_delay = np.zeros((X))

# Save the best voxels on basis of their correlation and delay, and their corresponding weights and test fmri signals
index = 0
for num,tup in enumerate(top):
    
    # Retrieve correlation, array and voxel position and the corresponding delay for the best voxel
    arr_pos = tup[1]
    corr = tup[0][0]
    vox_pos = tup[0][1]
    delay = tup[0][2]
    if(not(vox_pos in top_pos)):
        
        # Save position, correlation, weigths, test_fmri and the delay count for the best X voxels
        top_pos[index] = vox_pos
        top_corr[index] = corr
        top_W[:,index] = delay_dict[delay]['W'][:,arr_pos]
        unfiltered_test_fmri = delay_dict[delay]['test_fmri'][:,arr_pos]
        top_delay[index] = delay
        
        # Filter the test_fmri by removing the TRs that are unavailable for the longest delay
        sliced_fmri_list = []
        current_len = 0
        for video_file in test_video_list:
            
            # Determine current len of video based on the TRs of this delay
            video_len = video_TR_lengths[video_file]-2*delay
            
            # Determine difference with current delay and the largest delay and
            # filter the TRs that are unavailable for the largest delay
            delay_diff = delays[-1]-delay
            sliced_fmri = unfiltered_test_fmri[current_len+delay_diff:current_len+video_len-delay_diff]
            
            # Save sliced fmri and shift current len forward
            sliced_fmri_list.append(sliced_fmri)
            current_len = current_len + video_len
        
        # Concatenate all sliced fmri signals per video to one signal again
        top_test_fmri[:,index] = np.concatenate(sliced_fmri_list).ravel()
        index = index+1
        
    if(index == X):
        break

# Save dictionary to disk with the top weight matrix, top test fmri signal and top voxel positions
voxel_selection = {'W': top_W, 'test_fmri': top_test_fmri, 'pos': top_pos, 'corr': top_corr, 'delay': top_delay}
hdf5storage.savemat(save_name,voxel_selection)

# Output the distribution over what delays were chosen how many times
unique, counts = np.unique(top_delay, return_counts=True)
for delay,count in zip(unique,counts):
    print("The delay of " + str(delay) + " was chosen: " + str(count) + " times.")