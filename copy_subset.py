from yaml import load, Loader
from sys import argv
from os import walk, path, makedirs
import os
import shutil

yaml_file = argv[1]
dirpath = argv[2]
metadata = load(open(yaml_file), Loader=Loader)["rosbag2_bagfile_information"]
start_time = metadata["starting_time"]["nanoseconds_since_epoch"]
end_time = str(start_time + metadata["duration"]["nanoseconds"])
start_time = str(start_time)

start_time = start_time[:-9] + "." + start_time[-9:] + ".jpg"
end_time = end_time[:-9] + "." + end_time[-9:] + ".jpg"

_, _, filenames = next(walk(path.join(dirpath, "sequence")))
files_to_copy = []
for file in filenames:
    if file > start_time and file < end_time:
        files_to_copy.append(file)

files_to_copy = sorted(files_to_copy)
makedirs(path.join("oxford", "05", argv[1], "sequence"), exist_ok=True)
shutil.copy(path.join(dirpath, "ground_truth.txt"), path.join("oxford", "05", argv[1]))
for file in files_to_copy:
    shutil.copy(path.join(dirpath, "sequence", file), path.join("oxford", "05", argv[1], "sequence"))

os.remove(yaml_file)