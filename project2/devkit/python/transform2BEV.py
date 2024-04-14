#!/usr/bin/env python3

# THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
# Copyright (C) 2013
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
# UNPUBLISHED PROPRIETARY MATERIAL. ALL RIGHTS RESERVED.
# Authors: Tobias Kuehnl <tkuehnl@cor-lab.uni-bielefeld.de>
#           Jannik Fritsch <jannik.fritsch@honda-ri.de>

import os
import sys
from glob import glob
import cv2  # OpenCV
from BirdsEyeView import BirdsEyeView


def main(data_files: str, path_to_calib: str, output_path: str, calib_end: str = '.txt') -> None:
    """
    Main function for transforming images to BirdsEyeView (BEV).
    
    Args:
        data_files: Input file pattern to be transformed to BirdsEyeView, e.g., "/home/user/kitti_road/data/*.png".
        path_to_calib: Directory containing calibration data files, e.g., "/home/user/kitti_road/calib/".
        output_path: Directory where the BirdsEyeView data will be saved, e.g., "/home/user/kitti_road/data_bev".
        calib_end: File extension of calibration files. Defaults to '.txt'.
    """
    # Verify that the input data directory and calibration directory exist
    path_to_data = os.path.dirname(data_files)
    if not os.path.isdir(path_to_data):
        raise FileNotFoundError(f"The directory containing the input data does not exist: {path_to_data}")
    if not os.path.isdir(path_to_calib):
        raise FileNotFoundError(f"Calibration directory does not exist: {path_to_calib}")

    # Initialize the BirdsEyeView class
    bev = BirdsEyeView()

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get the list of data files
    file_list_data = glob(data_files)
    if not file_list_data:
        raise FileNotFoundError(f"Could not find files in: {path_to_data}")

    # Process each file
    for a_file in file_list_data:
        file_key = os.path.splitext(os.path.basename(a_file))[0]
        print(f"Transforming file {file_key} to BirdsEyeView")

        # Determine the calibration file
        calib_file = os.path.join(path_to_calib, file_key + calib_end)

        # Handle potential exceptions and fallback filenames
        if not os.path.isfile(calib_file):
            tags = file_key.split('_')
            if len(tags) == 3:
                calib_file = os.path.join(path_to_calib, f"{tags[0]}_{tags[2]}{calib_end}")

            # Raise an error if the calibration file still doesn't exist
            if not os.path.isfile(calib_file):
                raise FileNotFoundError(
                    f"Cannot find calibration file: {calib_file}. "
                    "Input data and calibration files are expected to have the same name (with different extensions)."
                )

        # Set up BirdsEyeView with the calibration file
        bev.setup(calib_file)

        # Read the input image
        data = cv2.imread(a_file, cv2.IMREAD_UNCHANGED)

        # Compute the BirdsEyeView
        data_bev = bev.compute(data)

        # Output file path
        fn_out = os.path.join(output_path, os.path.basename(a_file))

        # Write the output image (BEV)
        cv2.imwrite(fn_out, data_bev)
        print("Transformation done.")

    print(f"BirdsEyeView was stored in: {output_path}")


if __name__ == "__main__":
    # Check for the correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python transform2BEV.py <InputFiles> <PathToCalib> <OutputPath>")
        print(
            "<InputFiles>: Files you want to transform to BirdsEyeView, e.g., /home/user/kitti_road/data/*.png"
        )
        print(
            "<PathToCalib>: Directory containing calibration data, e.g., /home/user/kitti_road/calib/"
        )
        print(
            "<OutputPath>: Directory where the BirdsEyeView data will be saved, e.g., /home/user/kitti_road/data_bev"
        )
        print(
            "Note: It is assumed that input data and calibration files have the same name (with different extensions)."
        )
        sys.exit(1)

    # Parse arguments
    data_files = sys.argv[1]
    path_to_calib = sys.argv[2]
    output_path = sys.argv[3]

    # Execute main function
    main(data_files, path_to_calib, output_path)
