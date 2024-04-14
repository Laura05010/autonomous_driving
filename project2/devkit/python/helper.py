#!/usr/bin/env python3

# THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK

import numpy as np
from typing import Tuple, Dict, List, Optional

def eval_exp(
    gt_bin: np.ndarray,
    cur_prob: np.ndarray,
    thresholds: np.ndarray,
    valid_map: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Performs basic pixel-based evaluation.
    
    :param gt_bin: Ground truth binary map (2D array).
    :param cur_prob: Current probability map (2D array).
    :param thresholds: Array of threshold values.
    :param valid_map: Optional valid map (2D array), defaults to None.
    :return: Tuple of false negatives, false positives, positive sample count, and negative sample count.
    """
    assert cur_prob.ndim == 2 and gt_bin.ndim == 2, 'Input arrays must be 2D'

    # Append -Inf and Inf to thresholds
    thresholds_ext = np.concatenate(([-np.inf], thresholds, [np.inf]))

    # Histogram of false negatives
    if valid_map is not None:
        fn_array = cur_prob[(gt_bin == True) & (valid_map == 1)]
    else:
        fn_array = cur_prob[gt_bin == True]
    fn_hist = np.histogram(fn_array, bins=thresholds_ext)[0]
    fn_cum = np.cumsum(fn_hist)
    fn = fn_cum[:len(thresholds)]

    # Histogram of false positives
    if valid_map is not None:
        fp_array = cur_prob[(gt_bin == False) & (valid_map == 1)]
    else:
        fp_array = cur_prob[gt_bin == False]
    fp_hist = np.histogram(fp_array, bins=thresholds_ext)[0]
    fp_cum = np.flipud(np.cumsum(np.flipud(fp_hist)))
    fp = fp_cum[1:1 + len(thresholds)]

    # Calculate positive and negative sample counts
    if valid_map is not None:
        pos_num = np.sum((gt_bin == True) & (valid_map == 1))
        neg_num = np.sum((gt_bin == False) & (valid_map == 1))
    else:
        pos_num = np.sum(gt_bin == True)
        neg_num = np.sum(gt_bin == False)

    return fn, fp, pos_num, neg_num

def px_eval_maximize_f_measure(
    total_pos_num: int,
    total_neg_num: int,
    total_fn: np.ndarray,
    total_fp: np.ndarray,
    thresholds: np.ndarray
) -> Dict[str, float]:
    """
    Calculate precision, recall, F-measure, and average precision.
    
    :param total_pos_num: Total number of positive samples.
    :param total_neg_num: Total number of negative samples.
    :param total_fn: False negatives at different thresholds.
    :param total_fp: False positives at different thresholds.
    :param thresholds: Array of threshold values.
    :return: Dictionary of evaluation measures including average precision and maximum F-measure.
    """
    # Calculate true positives and true negatives
    total_tp = total_pos_num - total_fn
    total_tn = total_neg_num - total_fp
    
    # Validate true positives and true negatives
    valid = (total_tp >= 0) & (total_tn >= 0)
    assert valid.all(), 'Invalid values in evaluation'

    # Calculate recall and precision
    recall = total_tp / float(total_pos_num)
    precision = total_tp / (total_tp + total_fp + 1e-10)

    # Calculate average precision using Pascal VOC metric
    avg_prec = 0
    for i in np.arange(0, 1.1, 0.1):
        idx = np.where(recall >= i)
        max_prec = precision[idx].max()
        avg_prec += max_prec
    avg_prec /= 11.0

    # Calculate F-measure
    beta = 1.0
    beta_sq = beta ** 2
    f_measure = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + 1e-10)
    max_f_idx = f_measure.argmax()
    max_f = f_measure[max_f_idx]

    # Calculate additional measures
    tp_bst = total_tp[max_f_idx]
    tn_bst = total_tn[max_f_idx]
    fp_bst = total_fp[max_f_idx]
    fn_bst = total_fn[max_f_idx]

    # Prepare the result dictionary
    result_dict: Dict[str, float] = {
        'MaxF': max_f,
        'AvgPrec': avg_prec,
        'TP': tp_bst,
        'TN': tn_bst,
        'FP': fp_bst,
        'FN': fn_bst,
        'Precision': precision[max_f_idx],
        'Recall': recall[max_f_idx],
        'BestThresh': thresholds[max_f_idx],
    }

    return result_dict

def calc_eval_measures(eval_dict: np.ndarray, tag: str = '_wp') -> Dict[str, np.ndarray]:
    """
    Calculate evaluation measures based on the input evaluation dictionary.
    
    :param eval_dict: Evaluation dictionary containing TP, TN, FP, and FN values.
    :param tag: Optional tag for labeling results, defaults to '_wp'.
    :return: Dictionary of calculated measures including TPR, FPR, precision, recall, and others.
    """
    tp = eval_dict[:, 0].astype(np.float32)
    tn = eval_dict[:, 1].astype(np.float32)
    fp = eval_dict[:, 2].astype(np.float32)
    fn = eval_dict[:, 3].astype(np.float32)
    
    q = tp / (tp + fp + fn)
    p = tp + fn
    n = tn + fp
    
    tpr = tp / p
    fpr = fp / n
    fnr = fn / p
    tnr = tn / n
    accuracy = (tp + tn) / (p + n)
    precision = tp / (tp + fp)
    recall = tp / p

    # Prepare the output dictionary
    measures = {
        f'TP{tag}': tp,
        f'FP{tag}': fp,
        f'FN{tag}': fn,
        f'TN{tag}': tn,
        f'Q{tag}': q,
        f'A{tag}': accuracy,
        f'TPR{tag}': tpr,
        f'FPR{tag}': fpr,
        f'FNR{tag}': fnr,
        f'PRE{tag}': precision,
        f'REC{tag}': recall,
    }

    return measures
