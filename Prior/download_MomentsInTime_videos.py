# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:14:05 2019

@author: Leonieke

Script to download the videos in the Moments In Time dataset and save 0.7 seconds 
per video in 112x112 to disk in a FPS of 22.86. The parameters of the function 
aretwo row numbers, from and to which rows, in order to be able to run mulitple 
instances at the same time. The CSV used was obtained through the website of
the Moments in Time dataset on http://moments.csail.mit.edu/, but can also be 
found in the GitHub repository.
"""

import os
import sys
import csv
import time
import errno
import signal
import urllib
import moviepy.editor as mp
from itertools import islice

# Create classes and a signal function to be able to timeout the code
class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
    
signal.signal(signal.SIGALRM, timeout_handler) # Change the behavior of SIGALRM

# Get parameters from call
from_url = int(sys.argv[1]) # Set from which url number to start as the first parameter in command line call
to_url = int(sys.argv[2]) # Set at which url number to end as the second parameter in command line call

# Set changable parameters
save_path = os.getcwd() + "/Clips"

# Check if save directory already exists, else make it
if not(os.path.isdir(os.path.join(os.getcwd(), save_path))):
    os.makedirs(os.path.join(os.getcwd(), save_path))

# Create list to save the mit ids in from reading the csv file
mit_id_list = []

# Read the mit ids from the csv file
csv_file = os.getcwd() + "/MIT_ids.csv"
with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in islice(reader,from_url,to_url):
        mit_id_list.append(line[0].split(',')[0])
        
# Create all mit urls to download
url_basis = "http://data.csail.mit.edu/soundnet/actions3/"
urls = [url_basis + mit_id for mit_id in mit_id_list]

# Check save path to save clips to
if not(os.path.isdir(save_path)):
    os.makedirs(save_path)
    
# Download videos, convert to new size and fps, save 0.7 s clip and save to file
for num,url in enumerate(urls):
    print(num)
    # Start the timer. Once 45 seconds are over, download apparently failed, so a SIGALRM signal is sent.
    signal.alarm(45) 
    try:
        urllib.request.urlretrieve(url,save_path + "/" + str(num+from_url) + "-mit_full.mp4")
        clip = mp.VideoFileClip(save_path + "/" + str(num+from_url) + "-mit_full.mp4")
        
        # Resize video to 112x112
        if(clip.size[0] > clip.size[1]):
            clip = clip.fx(mp.vfx.resize, height=112)
            crop_left = (clip.size[0]-112)/2
            crop_right = clip.size[0] - crop_left
            clip = clip.crop(x1=crop_left,y1=0,x2=crop_right,y2=112)
        else:
            clip = clip.fx(mp.vfx.resize, width=112)
            crop_bot = (clip.size[1]-112)/2
            crop_top = clip.size[1] - crop_bot
            clip = clip.crop(x1=0,y1=crop_bot,x2=112,y2=crop_top)
        
        # Select a subclip from video
        sub_clip = clip.subclip(1,1.7)
        sub_clip_path = save_path + "/" + str(num+from_url) + "-mit.mp4"
        sub_clip.write_videofile(sub_clip_path,codec='libx264',audio=False,fps=22.86) # Save clip to disk without audio in preferred fps
        
        # Close all clip parameters
        sub_clip.reader.close()
        del sub_clip.reader
        if sub_clip.audio != None:
            sub_clip.audio.reader.close_proc()
            del sub_clip.audio
        del sub_clip

        # Close the original file in moviepy
        clip.reader.close()
        del clip.reader
        if clip.audio != None:
            clip.audio.reader.close_proc()
            del clip.audio
        del clip
        time.sleep(1) # Need to sleep 1 second to close all processes
        os.remove(save_path + "/" + str(num+from_url) + "-mit_full.mp4") # Remove original file from disk
    
    except TimeoutException:
        continue # continue the for loop if handling the video takes more than 45 seconds

    except KeyboardInterrupt:
        sys.exit() # Exit when ctrl+c is pressed
        
    except OSError as error:
        if error.errno == errno.ENOSPC:
            sys.exit() # Exit if disk is full or file can't be written
        else:
            continue
        
    except:
        continue # Continue to next video when other errors occur (e.g. video not found, download not completed etc.)
    
    else:
        # Reset the alarm
        signal.alarm(0)