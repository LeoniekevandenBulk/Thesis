# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:09:00 2019

@author: Leonieke

This script combines all the created top 100 prior videos dictionaries into one
final top 100 dictionary
"""
import os
import pickle

# Set directory of the location of the dictionaries
dict_dir = "ME_Top_100_Prior_Dictionaries"
dict_path = os.path.join(os.getcwd(),dict_dir)

# Set save name of the final dictionary
save_name = "ME_top_100_best_overall_clips.pkl"

# List dictionary files
dict_list = os.listdir(dict_path)

# Load the first dictionary of the list as a base
first_dict = pickle.load(open(os.path.join(dict_path,dict_list[0]), 'rb'), encoding='latin1')

# Initialize a new top 100 dictionary with the same layout
final_top_clips = {}
for TR in range(len(first_dict)):
    top = []
    for i in range(100):
        top.append((-1,''))
    final_top_clips[str(TR)] = top

# Loop over all dictionaries
for dict_name in dict_list:
    
    # Open dictionary from list
    dictionary = pickle.load(open(os.path.join(dict_path,dict_name), 'rb'), encoding='latin1')
    
    # Loop over all TRs to decide the best top 100 per TR
    for TR in dictionary:
        
        # Loop over all entries in this TR
        for clip in range(len(dictionary[TR])):
            
            # Check correlation and compare it to current lowest correlation in 
            # the final dictionary, if higher it gets added in
            corr = dictionary[TR][clip][0]
            if(corr > final_top_clips[TR][99][0]):
                clip_name = dictionary[TR][clip][1]
                final_top_clips[TR][99] = (corr,clip_name)
                final_top_clips[TR] = sorted(final_top_clips[TR],reverse=True)
                
#Save the dictionary containing the top 100 clips per TR per test video to disk
save_path = os.path.join(dict_path,save_name)
pkl_file = open(save_path,'wb')
pickle.dump(final_top_clips,pkl_file,protocol=2)
pkl_file.close()   