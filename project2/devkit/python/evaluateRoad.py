#!/usr/bin/env python3

# THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK

import sys
import os
from glob import glob
import numpy as np
import cv2  # OpenCV
from typing import List, Tuple, Optional, Dict

# Importing custom functions from the helper module
from helper import evalExp, pxEval_maximizeFMeasure

class DataStructure:
    """
    Class containing definitions for data paths and evaluation properties.
    """
    cats = ['um_lane', 'um_road', 'umm_road', 'uu_road']
    calib_end = '.txt'
    im_end = '.png'
    gt_end = '.png'
    prob_end = '.png'
    eval_property_list = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp', 'FPR_wp', 'FNR_wp']

def main(result_dir: str, train_dir: str, debug: bool = False) -> bool:
    """
    Main method for evaluating the road data.

    :param result_dir: Directory with the result probability maps, e.g., /home/elvis/kitti_road/my_results
    :param train_dir: Training directory (must contain gt_image_2), e.g., /home/elvis/kitti_road/training
    :param debug: Debug flag (optional)
    :return: True if evaluation succeeded for any category, False otherwise.
    """
    print("Starting evaluation...")
    print(f"Available categories are: {DataStructure.cats}")

    # Define the threshold array
    thresh = np.arange(0, 256) / 255.0
    train_data_subdir_gt = 'gt_image_2/'
    gt_dir = os.path.join(train_dir, train_data_subdir_gt)

    # Check that the result directory exists
    if not os.path.isdir(result_dir):
        print(f"Cannot find result_dir: {result_dir}")
        return False

    # Initialize results
    prob_eval_scores: List[Dict] = []
    eval_cats: List[str] = []

    # Evaluate each category
    for cat in DataStructure.cats:
        print(f"Executing evaluation for category {cat}...")
        fn_search = f"{cat}*{DataStructure.gt_end}"
        gt_file_list = glob(os.path.join(gt_dir, fn_search))
        if not gt_file_list:
            print(f"Error reading ground truth for category {cat}")
            continue

        # Initialize data for the category
        category_ok = True
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        total_pos_num = 0
        total_neg_num = 0

        # Evaluate files in the current category
        for fn_cur_gt in gt_file_list:
            file_key = os.path.splitext(os.path.basename(fn_cur_gt))[0]

            if debug:
                print(f"Processing file: {file_key}")

            # Read GT
            cur_gt = cv2.imread(fn_cur_gt, cv2.IMREAD_GRAYSCALE) > 0

            # Read probability map
            fn_cur_prob = os.path.join(result_dir, f"{file_key}{DataStructure.prob_end}")
            if not os.path.isfile(fn_cur_prob):
                print(f"Cannot find file: {fn_cur_prob} for category {cat}.")
                category_ok = False
                break
            
            cur_prob = cv2.imread(fn_cur_prob, cv2.IMREAD_GRAYSCALE)
            if cur_prob is None:
                print(f"Error: Could not read probability map {fn_cur_prob}")
                continue
            
            # Normalize the probability map
            cur_prob = np.clip((cur_prob.astype(np.float32)) / np.iinfo(cur_prob.dtype).max, 0.0, 1.0)

            # Evaluate the file
            fn, fp, pos_num, neg_num = evalExp(cur_gt, cur_prob, thresh, valid_map=None)

            # Validate evaluation results
            assert fn.max() <= pos_num, 'BUG: Positive samples exceeded'
            assert fp.max() <= neg_num, 'BUG: Negative samples exceeded'

            # Accumulate results
            total_fp += fp
            total_fn += fn
            total_pos_num += pos_num
            total_neg_num += neg_num

        if category_ok:
            print("Computing evaluation scores...")
            # Compute evaluation scores
            scores = pxEval_maximizeFMeasure(total_pos_num, total_neg_num, total_fn, total_fp, thresh=thresh)
            prob_eval_scores.append(scores)
            eval_cats.append(cat)

            # Output results for the category
            factor = 100
            for prop in DataStructure.eval_property_list:
                print(f"{prop}: {scores[prop] * factor:.2f}")

            print(f"Finished evaluating category: {eval_cats[-1]}")

    if eval_cats:
        print(f"Successfully finished evaluation for {len(eval_cats)} categories: {eval_cats}")
        return True
    else:
        print("No categories have been evaluated!")
        return False


#########################################################################
# Evaluation script entry point
#########################################################################
if __name__ == "__main__":
    # Check for the correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python evaluate_road.py <result_dir> <train_dir>")
        print("<result_dir> = Directory with the result probability maps, e.g., /home/elvis/kitti_road/my_results")
        print("<train_dir> = Training directory (must contain gt_image_2), e.g., /home/elvis/kitti_road/training")
        sys.exit(1)

    # Parse parameters
    result_dir = sys.argv[1]
    train_dir = sys.argv[2]

    # Execute main function
    success = main(result_dir, train_dir)
    
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
