import os

def sync_files(image_dir, ann_dir):
    # Get the list of image file names without the extension
    images = {os.path.splitext(file)[0] for file in os.listdir(image_dir) if file.endswith('.png')}
    
    # Counters for keeping track of deletions
    deleted_count = 0

    # Loop through all files in the annotation directory
    for ann_file in os.listdir(ann_dir):
        if ann_file.endswith('.txt'):
            ann_base = os.path.splitext(ann_file)[0]
            
            # Check if there's a corresponding image file
            if ann_base not in images:
                ann_file_path = os.path.join(ann_dir, ann_file)
                print(f"Deleting {ann_file_path} as no corresponding image exists.")
                os.remove(ann_file_path)
                deleted_count += 1
            else:
                print(f"Found valid image for {ann_file}")

    print(f"Total files deleted: {deleted_count}")

# Set the directories
image_directory = '/data/large-vehicle-train/val/images'
annotation_directory = '/data/large-vehicle-train/val/annfiles'

# Call the function
sync_files(image_directory, annotation_directory)
