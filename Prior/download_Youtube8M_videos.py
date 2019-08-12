# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:10:17 2018

@author: Leonieke

Script to download youtube videos from the Youtube 8M dataset and save 4 x 0.7 
seconds per video in 112x112 to disk in a FPS of 22.86. The parameters of the function are
two row numbers, from and to which rows, in order to be able to run mulitple 
instances at the same time. The CSV file was created by generate_Youtube8M_urls.py and 
can also be found in the GitHub repository.
"""

import os
import sys
import csv
import time
import errno
import signal
import numpy as np
import pytube
import moviepy.editor as mp
from itertools import islice

# Create classes and a signal function to be able to timeout the code
class TimeoutException(Exception): # Custom exception class
    pass

def timeout_handler(signum, frame): # Custom signal handler
    raise TimeoutException
    
signal.signal(signal.SIGALRM, timeout_handler) # Change the behavior of SIGALRM

## Get parameters from call
from_url = int(sys.argv[1]) # Set from which row number to start as the first parameter in command line call
to_url = int(sys.argv[2]) # Set at which row number to end as the second parameter in command line call

# Set directory to save clips
save_path = os.getcwd() + "/Clips"

# Check if save directory already exists, else make it
if not(os.path.isdir(os.path.join(os.getcwd(), save_path))):
    os.makedirs(os.path.join(os.getcwd(), save_path))

# Create list to save the youtube ids in from reading the csv file
youtube_id_list = []

# Read the youtube ids from the csv file
csv_file = os.getcwd() + "/Youtube8M_ids.csv"
with open(csv_file,'r') as csvfile:
    reader = csv.reader(csvfile)
    for youtube_id in islice(reader,from_url,to_url):
        youtube_id_list.append(youtube_id[0])

# Create all youtube urls to download
url_basis = "https://www.youtube.com/watch?v="
urls = [url_basis + youtube_id for youtube_id in youtube_id_list]

# Check save path to save clips to
if not(os.path.isdir(save_path)):
    os.makedirs(save_path)

# Download videos, convert to new size and fps, save one 0.7 s clips and save to file
for num,url in enumerate(urls):
    print(num)
    # Start the timer. Once 45 seconds are over, download apparently failed, so a SIGALRM signal is sent.
    signal.alarm(45) 
    try:
        yt = pytube.YouTube(url) # Download video from youtube
        video = yt.streams.filter(resolution='360p',subtype='mp4').first()
        video.download(output_path=save_path, filename=str(num+from_url)) # Save youtube video to disk
        clip = mp.VideoFileClip(save_path + "/" + str(num+from_url) + ".mp4")
        
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
            
        # Select 4 different subclips from video
        length = clip.duration
        clip_seconds = np.random.randint(length, size=4) # Determine random seconds from the video to turn into clips
        for sec in clip_seconds:
            sub_clip = clip.subclip(sec,sec+0.7)
            sub_clip_path = save_path + "/" + str(num+from_url) + "-" + str(sec) + ".mp4"
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
        os.remove(save_path + "/" + str(num+from_url) + ".mp4") # Remove original file from disk
    
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