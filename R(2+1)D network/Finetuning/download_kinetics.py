# -*- coding: utf-8 -*-
"""
@author: Leonieke van den Bulk

Adapted from the ActivityNet Repository (https://github.com/activitynet/ActivityNet/),
which is the official repository for the Kinetics dataset, which is where the csv
files come from as well.
"""

import os
import sys
import time
import errno
import signal
import pytube
import moviepy.editor as mp
from collections import OrderedDict
import pandas as pd

# Create classes and a signal function to be able to timeout the code
class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

def create_video_folders(dataset, kinetics_path):
    """Creates a directory for each label name in the dataset."""
    if 'label-name' not in dataset.columns:
        this_dir = os.path.join(kinetics_path, 'test')
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        # I should return a dict but ...
        return this_dir
    if not os.path.exists(kinetics_path):
        os.makedirs(kinetics_path)

    label_to_dir = {}
    for label_name in dataset['label-name'].unique():
        label_name = label_name.replace(" ", "_")
        this_dir = os.path.join(kinetics_path, label_name)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        label_to_dir[label_name] = this_dir
    return label_to_dir


def construct_video_filename(row, label_to_dir, trim_format='%06d'):
    """Given a dataset row, this function constructs the
       output filename for a given video.
    """
    basename = '%s_%s_%s.mp4' % (row['video-id'],
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    if not isinstance(label_to_dir, dict):
        dirname = label_to_dir
    else:
        dirname = label_to_dir[row['label-name'].replace(" ", "_")]
    return (dirname,basename)


def download_clip(video_identifier, dirname, basename,
                  start_time, end_time,
                  url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.
    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(dirname, str), 'output_filename must be string'
    assert len(video_identifier) == 11, 'video_identifier must have length 11'
    
    #print(dirname)
    #print(basename)
    signal.alarm(45) 
    try:
        yt = pytube.YouTube(url_base + video_identifier) # Download video from youtube
        video = yt.streams.filter(resolution='360p',subtype='mp4').first()
        video.download(output_path=dirname, filename="total_" + basename.split("mp4")[0]) # Save youtube video to disk
        clip = mp.VideoFileClip(dirname + "/" + "total_" + basename)
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
    
        sub_clip = clip.subclip(start_time,end_time)
        sub_clip_path = dirname + "/" + basename
        sub_clip.write_videofile(sub_clip_path,codec='libx264',audio=False, fps=22.86) # Save clip to disk without audio in preferred fps
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
        os.remove(dirname + "/" + "total_" + basename) # Remove original file from disk
    
    except TimeoutException:
        pass # continue the loop if handling the video takes more than 45 seconds

    except KeyboardInterrupt:
        sys.exit() # Exit when ctrl+c is pressed
        
    except OSError as error:
        if error.errno == errno.ENOSPC:
            sys.exit() # Exit if disk is full or file can't be written
        else:
            pass
        
    except:
        pass # Continue to next video when other errors occur (e.g. video not found, download not completed etc.)
    
    else:
        # Reset the alarm
        signal.alarm(0)

def download_clip_wrapper(kinetics_path, row, label_to_dir, trim_format):
    """Wrapper for parallel processing purposes."""
    (dirname,basename) = construct_video_filename(row, label_to_dir,
                                               trim_format)

    download_clip(row['video-id'], dirname, basename,
                                    row['start-time'], row['end-time'])


def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
    """Returns a parsed DataFrame.
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'
    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
    if 'youtube_id' in df.columns:
        columns = OrderedDict([
            ('youtube_id', 'video-id'),
            ('time_start', 'start-time'),
            ('time_end', 'end-time'),
            ('label', 'label-name')])
        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]
    return df


def main(input_csv, kinetics_path,
         trim_format='%06d',
         drop_duplicates=False):

    # Reading and parsing Kinetics.
    dataset = parse_kinetics_annotations(input_csv)

    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, kinetics_path)

    # Download all clips.
    for i, row in dataset.iterrows():
        download_clip_wrapper(kinetics_path, row, label_to_dir, trim_format)



if __name__ == '__main__':
    # Change the behavior of SIGALRM for exceptions
    signal.signal(signal.SIGALRM, timeout_handler) 
    
    # Load the train set
    input_csv_train =  os.path.join(os.getcwd(), "kinetics-400_train.csv")
    kinetics_path_train = os.path.join(os.getcwd(), "kinetics/train")
    main(input_csv_train, kinetics_path_train)
    
    # Load the validation set
    input_csv_val =  os.path.join(os.getcwd(), "kinetics-400_val.csv")
    kinetics_path_val = os.path.join(os.getcwd(), "kinetics/val")
    main(input_csv_val, kinetics_path_val)