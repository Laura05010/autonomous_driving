#!/usr/bin/env python3

# THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
# File: simpleExample_transformTestResults2BEV.py
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
from computeBaseline import main as compute_baseline_main
from transform2BEV import main as transform_to_bev_main


def main():
    """
    Main function to process testing data in the perspective domain and 
    transform the results to the metric BEV.
    """
    # Check for the correct number of arguments
    if len(sys.argv) < 2:
        print("Usage: python simpleExample_transformTestResults2BEV.py <datasetDir> <outputDir>")
        print(
            "<datasetDir> = Base directory of the KITTI Road benchmark dataset (should contain training and testing),"
            " e.g., /home/user/kitti_road/"
        )
        print(
            "<outputDir> = Directory to save the baseline results, e.g., /home/user/kitti_road/results/"
        )
        sys.exit(1)

    # Parse parameters
    dataset_dir = sys.argv[1]
    if not os.path.isdir(dataset_dir):
        print(f"Error: <datasetDir>={dataset_dir} does not exist.")
        sys.exit(1)

    # Determine output directory
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        # Default output directory
        output_dir = os.path.join(dataset_dir, 'results')

    # Define paths
    test_data_path_to_calib = os.path.join(dataset_dir, 'testing/calib')
    output_dir_perspective = os.path.join(output_dir, 'baseline_perspective_test')
    output_dir_bev = os.path.join(output_dir, 'baseline_bev_test')

    # Run computeBaseline script to generate example classification results on testing set
    # Replace this with your algorithm to generate real results
    train_dir = os.path.join(dataset_dir, 'training')
    test_dir = os.path.join(dataset_dir, 'testing')
    compute_baseline_main(train_dir, test_dir, output_dir_perspective)

    # Convert baseline in perspective space into BEV space
    # If your algorithm provides results in perspective space,
    # you need to run this script before submission!
    input_files = os.path.join(output_dir_perspective, '*.png')
    transform_to_bev_main(input_files, test_data_path_to_calib, output_dir_bev)

    # Note: Now you need to zip the contents of the directory 'outputDir_bev' and upload
    # the zip file to the KITTI server.


if __name__ == "__main__":
    main()
