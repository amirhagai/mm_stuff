import os
import re

def check_image_seg_count(base_path):
    # This dictionary will store directories with mismatched counts
    mismatched_folders = {}
    i=0
    for folder in os.listdir(base_path):
        i+=1

        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)

            pattern_images = re.compile(r'^-?\d+(\.\d+)?\.png$')
            matching_images = [file for file in files if re.match(pattern_images, file)]

            pattern_segs = re.compile(r'^-?\d+(\.\d+)?_seg\.png$')
            matching_segs = [file for file in files if re.match(pattern_segs, file)]

            if len(matching_images) != len(matching_segs):
                mismatched_folders[folder] = (len(matching_images), len(matching_segs))
    print(i)
    return mismatched_folders

# Replace this with your actual base directory path
base_path = "/data/split_ss_dota/train_injected_container/mid_reults"
mismatches = check_image_seg_count(base_path)

# Print out the folders with mismatches and their counts
for folder, counts in mismatches.items():
    print(f"Folder {folder} has {counts[0]} images and {counts[1]} segmentation files, which do not match.")