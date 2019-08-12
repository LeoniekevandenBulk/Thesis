import os
import sys

# Script to write all the names of the files in a folder to a .txt
# First argument should be the folder of interest, second argument the name of the new .txt file

basedir = os.getcwd()
to_file_dir = sys.argv[1] 
files = os.listdir(os.path.join(basedir, to_file_dir))
txt_name = sys.argv[2]
with open(txt_name,'w') as file_dir:
    for f in files:
        file_dir.write(f + '\n')
