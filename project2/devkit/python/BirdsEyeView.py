#!/usr/bin/env python

import numpy as np
from glob import glob
import os
import sys
import cv2
from typing import Tuple, List, Any

class DataStructure:
    """
    All the definitions go in here.
    """
    cats = ['um_lane', 'um_road', 'umm_road', 'uu_road']
    calib_end = '.txt'
    im_end = '.png'
    gt_end = '.png'
    prob_end = '.png'
    eval_property_list = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp', 'FPR_wp', 'FNR_wp']
    train_data_subdir_gt = 'gt_image_2'
    test_data_subdir_im2 = 'image_2'
    image_shape_max: Tuple[int, int] = (376, 1242)


def compute_baseline(train_dir: str, test_dir: str, output_dir: str) -> None:
    """
    Computes the location potential as a simple baseline for classifying the data.

    :param train_dir: Directory of training data (has to contain ground truth data).
    :param test_dir: Directory with testing data (has to contain images).
    :param output_dir: Directory where the baseline results will be saved.
    """
    train_data_path_gt = os.path.join(train_dir, DataStructure.train_data_subdir_gt)

    print(f"Computing category-specific location potential as a simple baseline for classifying the data...")
    print(f"Using ground truth data from: {train_data_path_gt}")
    print(f"All categories = {DataStructure.cats}")

    # Loop over all categories
    for cat in DataStructure.cats:
        cat_tags = cat.split('_')
        print(f"Computing on dataset: {cat_tags[0]} for class: {cat_tags[1]}")

        train_data_files_gt = glob(os.path.join(train_data_path_gt, cat + '*' + DataStructure.gt_end))
        train_data_files_gt.sort()

        if not train_data_files_gt:
            print(f"Error: Cannot find ground truth data in {train_data_path_gt}. Skipping category {cat}.")
            continue
        
        # Compute location potential
        location_potential = np.zeros(DataStructure.image_shape_max, dtype=np.float32)

        # Loop over all ground truth files for the particular category
        for train_data_file_gt in train_data_files_gt:
            gt_image = cv2.imread(train_data_file_gt, cv2.IMREAD_GRAYSCALE)
            if gt_image is None:
                print(f"Error: Could not read ground truth image: {train_data_file_gt}.")
                continue
            
            # Convert the image to binary (true for road, false otherwise)
            gt_image = gt_image > 0
            
            # Add the ground truth data to the location potential
            location_potential[:gt_image.shape[0], :gt_image.shape[1]] += gt_image
        
        # Normalize location potential
        location_potential /= len(train_data_files_gt)

        # Convert to uint8 and scale to 255
        location_potential_u8 = (location_potential * 255).astype(np.uint8)
        
        print(f"Done: computing location potential for category: {cat}.")

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
        # Process testing data files
        test_data_files_im2 = glob(os.path.join(test_dir, DataStructure.test_data_subdir_im2, f"{cat_tags[0]}_*{DataStructure.im_end}"))
        test_data_files_im2.sort()
        
        print(f"Writing location potential as perspective probability map into {output_dir}.")
        
        for test_data_file_im2 in test_data_files_im2:
            # Extract the filename
            file_name_im2 = os.path.basename(test_data_file_im2)
            ts_str = file_name_im2.split(f"{cat_tags[0]}_")[-1]
            
            # Construct the output filename
            output_filename = os.path.join(output_dir, cat + ts_str)
            
            # Write the location potential to the output file
            cv2.imwrite(output_filename, location_potential_u8)
        
        print(f"Done: Creating perspective baseline for category: {cat}.")


if __name__ == "__main__":
    # Check for the correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python compute_baseline.py <TrainDir> <TestDir> <OutputDir>")
        print("<TrainDir> = Directory of training data (must contain ground truth data: gt_image_2).")
        print("<TestDir> = Directory of testing data (must contain images: image_2).")
        print("<OutputDir> = Directory where the baseline results will be saved.")
        sys.exit(1)
    
    # Parse parameters
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Execute main function
    compute_baseline(train_dir, test_dir, output_dir)
