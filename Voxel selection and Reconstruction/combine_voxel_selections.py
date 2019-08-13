# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:41:05 2019

@author: Leonieke

Script to combine two voxel selections (made by voxel_selection.py) from two 
different domains (in this case Motion Energy and the pooling layer from a 
action recognition neural network) into one combined voxel selection.
"""

import hdf5storage
import numpy as np

# Set the encoding model files to be combined
ME_encoding_model_file = "ME_voxel_selection_alldelays_2000.mat"
NN_encoding_model_file = "NN_Pool_voxel_selection_alldelays_2000.mat"

# Set the save name for the combined model
save_name = "combined_voxel_selection_alldelays_2700.mat"

# Load the encoding models
ME_encoding_model = hdf5storage.loadmat(ME_encoding_model_file)
ME_voxel_positions = ME_encoding_model['pos']
ME_corrs = ME_encoding_model['corr']
ME_weights = ME_encoding_model['W']
ME_test_fmri = ME_encoding_model['test_fmri']
ME_delays = ME_encoding_model['delay']

NN_encoding_model = hdf5storage.loadmat(NN_encoding_model_file)
NN_voxel_positions = NN_encoding_model['pos']
NN_corrs = NN_encoding_model['corr']
NN_weights = NN_encoding_model['W']
NN_test_fmri = NN_encoding_model['test_fmri']
NN_delays = NN_encoding_model['delay']

# Find the voxels that are in both models and the voxels that are only in one of the models
intersection = np.intersect1d(ME_voxel_positions,NN_voxel_positions)
ME_unique = np.setdiff1d(ME_voxel_positions,NN_voxel_positions)
NN_unique = np.setdiff1d(NN_voxel_positions,ME_voxel_positions)
print(len(ME_unique))
print(len(NN_unique))
# Set the new amount of voxels in the model and new matrices to save them
nr_of_voxels = len(intersection) + len(ME_unique) + len(NN_unique)
new_pos = np.zeros((nr_of_voxels))
new_corr = np.zeros((nr_of_voxels))
model = np.zeros((nr_of_voxels))
delay = np.zeros((nr_of_voxels))
new_test_fmri = np.zeros((ME_test_fmri.shape[0],nr_of_voxels))
new_ME_W = np.zeros((ME_weights.shape[0],nr_of_voxels))
new_NN_W = np.zeros((NN_weights.shape[0],nr_of_voxels))

# Select from the voxels that are in both models, the voxel from the model with the highest correlation (if the correlation is higher than 0.3)
index = 0
ME_best = 0
NN_best = 0
for voxel in intersection:
    ME_index = np.where(ME_voxel_positions==voxel)[0][0]
    ME_corr = ME_corrs[ME_index]
    NN_index = np.where(NN_voxel_positions==voxel)[0][0]
    NN_corr = NN_corrs[NN_index]
    if(ME_corr > NN_corr):
        new_pos[index] = voxel
        new_corr[index] = ME_corr
        model[index] = 0
        delay[index] = ME_delays[ME_index]
        new_test_fmri[:,index] = ME_test_fmri[:,ME_index]
        new_ME_W[:,index] = ME_weights[:,ME_index]
        index = index+1
        ME_best += 1
    elif(NN_corr > ME_corr):
        new_pos[index] = voxel
        new_corr[index] = NN_corr
        model[index] = 1
        delay[index] = NN_delays[NN_index]
        new_test_fmri[:,index] = NN_test_fmri[:,NN_index]
        new_NN_W[:,index] = NN_weights[:,NN_index]
        index = index+1
        NN_best += 1

print(ME_best)
print(NN_best)
# Add the voxels that are just in one of the models if they have a correlation higher than 0.3
for voxel in ME_unique:
    ME_index = np.where(ME_voxel_positions==voxel)[0][0]
    new_pos[index] = voxel
    new_corr[index] = ME_corrs[ME_index]
    model[index] = 0
    delay[index] = ME_delays[ME_index]
    new_test_fmri[:,index] = ME_test_fmri[:,ME_index]
    new_ME_W[:,index] = ME_weights[:,ME_index]
    index = index+1

for voxel in NN_unique:
    NN_index = np.where(NN_voxel_positions==voxel)[0][0]
    #if(NN_corrs[NN_index] >= 0.3):
    new_pos[index] = voxel
    new_corr[index] = NN_corrs[NN_index]
    model[index] = 1
    delay[index] = NN_delays[NN_index]
    new_test_fmri[:,index] = NN_test_fmri[:,NN_index]
    new_NN_W[:,index] = NN_weights[:,NN_index]
    index = index+1

# Trim the matrices to the correct length
new_pos = new_pos[0:index]
new_corr = new_corr[0:index]
model = model[0:index]
delay = delay[0:index]
new_test_fmri = new_test_fmri[:,0:index]
new_ME_W = new_ME_W[:,0:index]
new_NN_W = new_NN_W[:,0:index]

# Sort new matrices such that the order is based on the correlation
sort_indices = np.argsort(new_corr)[::-1]
new_corr = np.asarray([new_corr[i] for i in sort_indices], dtype=np.float64)
new_pos = np.asarray([new_pos[i] for i in sort_indices], dtype=np.float64)
model = np.asarray([model[i] for i in sort_indices], dtype=np.int64)
delay = np.asarray([delay[i] for i in sort_indices], dtype=np.int64)
new_test_fmri = new_test_fmri.take(sort_indices,1)
new_ME_W = new_ME_W.take(sort_indices,1)
new_NN_W = new_NN_W.take(sort_indices,1)

# Save dictionary to disk with the top weight matrix, top test fmri signal and top voxel positions
new_voxel_selection = {'pos': new_pos, 'corr': new_corr, 'model': model, 'delay': delay, 'ME_W': new_ME_W, 'NN_W': new_NN_W, 'test_fmri': new_test_fmri}
hdf5storage.savemat(save_name,new_voxel_selection)

# Output the distribution over what delays were chosen how many times
unique, counts = np.unique(delay, return_counts=True)
for delay,count in zip(unique,counts):
    print("The delay of " + str(delay) + " was chosen: " + str(count) + " times.")