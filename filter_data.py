
import glob
import os.path as osp
from mmrotate.datasets.dota import DOTADataset
from pathlib import Path
import os
import shutil




def get_txt_files(data_type, data_root = '/data/split_ss_dota/'):
        d = DOTADataset(data_root = data_root,
                ann_file=f'{data_type}/annfiles/',
                data_prefix=dict(img_path=f'{data_type}/images/'),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=None)
        txt_files = glob.glob(osp.join(d.ann_file, '*.txt'))
        return txt_files





def create_symlinks(file_list, dest_folder, source_dirs):
    os.makedirs(dest_folder, exist_ok=True)  # Ensure destination directory exists
    for file_name in file_list:
        # Change .txt in file names to .png
        image_file_name = file_name.replace('.txt', '.png')
        pth_img = Path(image_file_name)
        # Check each source directory for the existence of the image
        for source_dir in source_dirs:
            source_path = Path(source_dir) / pth_img.name
            if source_path.exists():
                # Construct destination path
                dest_path = Path(dest_folder) / pth_img.name
                # Create a symlink at the destination pointing to the source file
                if not dest_path.exists():  # Check if symlink already exists
                    os.symlink(source_path, dest_path)
                    print(f"Symlink created for {source_path} at {dest_path}")
                else:
                    print(f"Symlink already exists for {dest_path}")
                break  # Stop checking once the symlink is created
            else:
                print(f"File not found: {source_path}")


def create_files_arr(txt_files, cls_type='large-vehicle'):
        relevance = []
        non_relevance = []
        for txt_file in txt_files:
                classes = set()
                with open(txt_file) as f:
                        s = f.readlines()
                        for si in s:
                                instance = {}
                                bbox_info = si.split()
                                cls_name = bbox_info[8]
                                classes.add(cls_name)
                if cls_type in classes:
                        relevance.append(txt_file)
                else:
                        non_relevance.append(txt_file)
        return relevance, non_relevance



def main():
        txt_files_train = get_txt_files('train')
        txt_files_val = get_txt_files('val')

        relevance_train, non_relevance_train = create_files_arr(txt_files_train)
        relevance_val, non_relevance_val = create_files_arr(txt_files_val)

        create_symlinks(relevance_train, '/data/large-vehicle-train',["/data/split_ss_dota/train/images/"])

        create_symlinks(non_relevance_train, '/data/without-large-vehicle-train', ["/data/split_ss_dota/train/images/"])

        create_symlinks(relevance_val, '/data/large-vehicle-val', ["/data/split_ss_dota/val/images/"])

        create_symlinks(non_relevance_val, '/data/without-large-vehicle-val', ["/data/split_ss_dota/val/images/"])

if __name__ == "__main__":

      main()