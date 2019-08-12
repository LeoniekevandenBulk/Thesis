# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:20:05 2019

@author: Leonieke

Script to determine the significance between the different reconstructions and
between the reconstructions and a random reconstruction.
"""

import pickle
import numpy as np
from scipy.stats import ranksums,wilcoxon

# Set the load names of the saved pickles containing the CW-SSIM values
ME_pkl = "ME_Alexnet_correlation_analysis.pkl"
NN_pkl = "NN_Alexnet_correlation_analysis.pkl"
combined_pkl = "Combined_Alexnet_correlation_analysis.pkl"
random_pkl = "Random_Alexnet_correlation_analysis.pkl"

# Set name of the .txt file to write the results to
result_txt = "Significances.txt"
file = open(result_txt,"w") 

# Load pickles
ME_corr = pickle.load(open(ME_pkl, 'rb'))
NN_corr = pickle.load(open(NN_pkl, 'rb'))
combined_corr = pickle.load(open(combined_pkl, 'rb'))
random_corr = pickle.load(open(random_pkl, 'rb'))
for layer in ME_corr:
    ME_corr_layer = ME_corr[layer]
    NN_corr_layer = NN_corr[layer]
    combined_corr_layer = combined_corr[layer]
    random_corr_layer = random_corr[layer]

    # Do a wilcoxon rank sum test to test whether the models perform significantly 
    # better than random
    ME_statistic, ME_pvalue = ranksums(random_corr_layer, ME_corr_layer)
    NN_statistic, NN_pvalue = ranksums(random_corr_layer, NN_corr_layer)
    combined_statistic, combined_pvalue = ranksums(random_corr_layer, combined_corr_layer)
    
    # Also do a wilcoxon signed rank test to test whether the models perform significantly 
    # better than the other models
    ME_NN_statistic, ME_NN_pvalue = wilcoxon(ME_corr_layer, NN_corr_layer)
    ME_combined_statistic, ME_combined_pvalue = wilcoxon(ME_corr_layer, combined_corr_layer)
    NN_combined_statistic, NN_combined_pvalue = wilcoxon(NN_corr_layer, combined_corr_layer)
    
    # Report p-values and means
    file.write(layer + "\n")
    file.write("\n")
    file.write("Mean of Random correlation: " + str(np.mean(random_corr_layer)) + "\n")
    file.write("Mean of ME correlation: " + str(np.mean(ME_corr_layer)) + "\n")
    file.write("Mean of NN correlation: " + str(np.mean(NN_corr_layer)) + "\n")
    file.write("Mean of Combined correlation: " + str(np.mean(combined_corr_layer)) + "\n")
    file.write("\n")
    file.write("Std of Random correlation: " + str(np.std(random_corr_layer)) + "\n")
    file.write("Std of ME correlation: " + str(np.std(ME_corr_layer)) + "\n")
    file.write("Std of NN correlation: " + str(np.std(NN_corr_layer)) + "\n")
    file.write("Std of Combined correlation: " + str(np.std(combined_corr_layer)) + "\n")
    file.write("\n")
    file.write("p-value of ME vs Random: " + str(ME_pvalue) + "\n")
    file.write("p-value of NN vs Random: " + str(NN_pvalue) + "\n")
    file.write("p-value of Combined vs Random: " + str(combined_pvalue) + "\n")
    file.write("\n")
    file.write("p-value of ME vs NN: " + str(ME_NN_pvalue) + "\n")
    file.write("p-value of ME vs Combined: " + str(ME_combined_pvalue) + "\n")
    file.write("p-value of NN vs Combined: " + str(NN_combined_pvalue) + "\n")
    file.write("\n")
    file.write("---------------\n")
    
file.close() 