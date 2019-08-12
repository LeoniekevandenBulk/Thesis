# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:27:31 2018

@author: Leonieke

General script to convert the hidden video ids from the Youtube 8M dataset to the actual youtube ids
"""

import os
import csv
import requests
from bs4 import BeautifulSoup
import concurrent.futures

# Define function to visit url and return contents
def load_url(url):
    request = requests.get(url)
    return request.text

# Make empty list to save hidden ids in
hidden_id_list = []

# Make empty list to save final youtube ids in
youtube_id_list = []

# Open train and validation labels file and save the hidden ids to list (labels comes from https://github.com/google/youtube-8m#ground-truth-label-files)
csv_file_train = os.getcwd() + "/Youtube8M_train_labels.csv"
csv_file_val = os.getcwd() + "/Youtube8M_validate_labels.csv"
with open(csv_file_train, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for id,labels in reader:
        hidden_id_list.append(id)
with open(csv_file_val, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for id,labels in reader:
        hidden_id_list.append(id)
		
# Loop through all hidden ids, visit javascript url and save original youtube id 
# Generate javascript urls from hidden ids (based on https://research.google.com/youtube8m/video_id_conversion.html)
url_basis = "http://data.yt8m.org/2/j/i/"
urls = [url_basis + hidden_id[0:2] + "/" + hidden_id + ".js" for hidden_id in hidden_id_list]

# Loop through all urls and send multithreaded requests
CONNECTIONS = 100
x=0
with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
    execute_url = (executor.submit(load_url, url) for url in urls)
    for completed in concurrent.futures.as_completed(execute_url):
        print(x)
        x = x+1
        try:
            url_data = completed.result()
            url_text = str(BeautifulSoup(url_data,features="lxml")) # Save webpage as Beautifulsoup object
            youtube_id = url_text.split(",")[1].split("\"")[1] # Split html on webpage to get the youtube_id
            youtube_id_list.append(youtube_id)
        except:
            pass

# Write all youtube_ids to new csv
save_path = os.getcwd() + "/Youtube8M_ids.csv"
with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for id in youtube_id_list:
        writer.writerow([id])
