# -*- coding: utf-8 -*-
"""
@author: Leonieke

Maps Caffe2 parameters in a numpy file to Pytorch for the R(2+1)D model
Created on Thu Nov 29 16:03:39 2018

"""
import numpy as np

def convert(path_caffe2_model):
    """
    Makes a dictionary containing the mapping from the Caffe2 parameters to the Pytorch parameters
    """
    # Initialize a mapping dictionary and load the caffe2 parameters
    mapping = {}
    caffe_dict = np.load(path_caffe2_model, encoding='bytes').view(np.recarray).item()
    
    # Already establish a dictionary for the difference in names between the attributes of the parameters
    attributes = {'w':'weight', 'b':'bias', 'rm':'running_mean', 'riv':'running_var', 's':'weight'}
    
    # Loop over all parameters in the caffe2 dictionary
    for param_name in caffe_dict:
        
        # Read the param name and split it in multiple parts to recognize its meaning easier
        name = param_name.decode('utf-8') # Decode byte string to string
        name_split = name.split('_')
        first_part = name_split[0]
        last_part = name_split[-1]
        
        # Initialize the corresponding pytorch parameter name
        pytorch_name = ""
        
        # Make the actual mapping from parameters in Caffe2 to Pytorch
        if(first_part == 'conv1'): # First set of spatial and temporal convolutions (including batch normalization)
            basis = 'res2plus1d.conv1.'

            if(name_split[1] == 'middle'): # Spatial convolution and batch normalization
                if(name_split[2] == 'spatbn'):
                    pytorch_name = basis + 'bn1.' + attributes[last_part]
                else:
                    pytorch_name = basis + 'spatial_conv.' + attributes[last_part]
            elif(name_split[1] == 'spatbn'): # Batch normalization after the temporal convolution
                pytorch_name = basis + 'bn2.' + attributes[last_part]
            else: # Temporal convolution
                pytorch_name = basis + 'temporal_conv.' + attributes[last_part]
        
        elif(first_part == 'comp'): # All other sets of spatial and temporal convolutions (including batch normalization)
            if(int(name_split[1])%2 == 0):
                basis = 'res2plus1d.conv{}.block1.'.format(int(np.floor(int(name_split[1])/2)+2))
            else:
                basis = 'res2plus1d.conv{}.blocks.0.'.format(int(np.floor(int(name_split[1])/2)+2))
                
            if(name_split[2] == 'conv'): # Convolutions
                if(name_split[4] == 'middle'): # Spatial convolution
                    pytorch_name = basis + 'conv{}.spatial_conv.'.format(name_split[3]) + attributes[last_part]
                else: # Temporal convolution
                    pytorch_name = basis + 'conv{}.temporal_conv.'.format(name_split[3]) + attributes[last_part]
            else: # Batch normalizations
                if(name_split[4] == 'middle'): # Batch normalization after the spatial convolution
                    pytorch_name = basis + 'conv{}.bn1.'.format(name_split[3]) + attributes[last_part]
                else: # Batch normalization after the temporal convolution
                    pytorch_name = basis + 'bn{}.'.format(name_split[3]) + attributes[last_part]
                    
        elif(first_part == 'shortcut'): # Downsample step
            basis = 'res2plus1d.conv{}.block1.'.format(int(np.floor(int(name_split[2])/2)+2))
    
            if(name_split[3] == 'spatbn'): # Batch normalization of the downsample step
                pytorch_name = basis + 'downsamplebn.' + attributes[last_part]
            else: # Convolution of the downsample step
                pytorch_name = basis + 'downsampleconv.' + attributes[last_part]
            
        elif(first_part == 'last'): # Fully connected layer
            pytorch_name = 'linear.' + attributes[last_part]

        # Add the found mapping to the mapping dictionary
        mapping[name] = pytorch_name

    # Return the mapping dictionary at the end
    return mapping