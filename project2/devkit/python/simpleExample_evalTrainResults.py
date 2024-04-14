#!/usr/bin/env python3

# THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
# File: simpleExample_evalTrainResults.py

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
from evaluateRoad import main as evaluate_road_main


def main():
    """
    Main function to evaluate training data in the perspective domain.
    """
    # Check if the correct number of arguments is provided
    if len(sys.argv) < 2:
        print(
            "Usage: python simpleExample_evalTrainResults.py <datasetDir> <outputDir>"
        )
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

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        # Default output directory
        output_dir = os.path.join(dataset_dir, 'results')

    # Set paths for training data and output directory for perspective results
    train_dir = os.path.join(dataset_dir, 'training')
    output_dir_perspective = os.path.join(output_dir, 'baseline_perspective_train')

    # Run computeBaseline script to generate example classification results on training set
    compute_baseline_main(train_dir, train_dir, output_dir_perspective)

    # Run evaluation script on perspective train data
    # Note: Final evaluation on server is done in BEV space and uses a 'valid_map'
    # indicating the BEV areas that are invalid (no correspondence in perspective space)
    evaluate_road_main(output_dir_perspective, train_dir)


if __name__ == "__main__":
    main()
