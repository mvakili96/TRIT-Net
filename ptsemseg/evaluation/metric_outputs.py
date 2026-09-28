"""Demo/eval metric-output writers.

These helpers intentionally preserve the legacy text filenames, write modes,
and summary prints used by the copied demo/eval runner.
"""

import os

import numpy as np


def save_demo_eval_metric_outputs(
    res_eval,
    metrics_output_dir,
    flag_save_data,
    data_in_use,
    flag_single_multiple_path_evaluation,
):
    """Write legacy demo/eval metric text files and summary logs."""

    sum_prec = 0
    sum_rec = 0

    sum_path_prec = 0
    sum_path_recall = 0

    sum_all_prec = 0
    sum_all_recall = 0

    sum_IoU_Class_0 = 0
    sum_IoU_Class_1 = 0
    sum_IoU_Class_2 = 0
    sum_IoU_Class_3 = 0
    sum_GT_3 = 0

    sum_time_net = 0
    sum_time_pp = 0

    if flag_save_data and data_in_use <= 3:
        with open(os.path.join(metrics_output_dir, 'precision_1.txt'), 'w') as f:
            for item in res_eval:
                prec = item["precision"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    f.write('%f' % prec)
                    f.write("\n")

                sum_prec = sum_prec + prec

                sum_IoU_Class_0 += item["Class_0"]
                sum_IoU_Class_1 += item["Class_1"]
                sum_IoU_Class_2 += item["Class_2"]
                sum_IoU_Class_3 += item["GT_3"] * item["Class_3"]
                sum_GT_3 += item["GT_3"]

                sum_path_prec += item["path_level_prec"]
                sum_path_recall += item["path_level_recall"]

                sum_all_prec += item["all_pixel_prec"]
                sum_all_recall += item["all_pixel_recall"]

                sum_time_net += item["time_net"]
                sum_time_pp += item["time_pp"]

        with open(os.path.join(metrics_output_dir, 'recall.txt'), 'w') as f:
            for item in res_eval:
                rec = item["recall"]
                f.write('%f' % rec)
                f.write("\n")
                sum_rec = sum_rec + rec

        with open(os.path.join(metrics_output_dir, 'TP.txt'), 'w') as f:
            for item in res_eval:
                TP = item["TP"]
                f.write('%d' % TP)
                f.write("\n")

        with open(os.path.join(metrics_output_dir, 'FP.txt'), 'w') as f:
            for item in res_eval:
                FP = item["FP"]
                f.write('%d' % FP)
                f.write("\n")

        with open(os.path.join(metrics_output_dir, 'FN.txt'), 'w') as f:
            for item in res_eval:
                FN = item["FN"]
                f.write('%d' % FN)
                f.write("\n")

        avg_precision = sum_prec / len(res_eval)
        avg_recall = sum_rec / len(res_eval)

        mIoU_Class_0 = sum_IoU_Class_0 / len(res_eval)
        mIoU_Class_1 = sum_IoU_Class_1 / len(res_eval)
        mIoU_Class_2 = sum_IoU_Class_2 / len(res_eval)
        mIoU_Class_3 = sum_IoU_Class_3 / sum_GT_3

        avg_path_precision = sum_path_prec / len(res_eval)
        avg_path_recall = sum_path_recall / len(res_eval)

        avg_all_precision = sum_all_prec / len(res_eval)
        avg_all_recall = sum_all_recall / len(res_eval)

        avg_time_net = sum_time_net / len(res_eval)
        avg_time_pp = sum_time_pp / len(res_eval)

        print("TP PIXEL LEVEL [AVERAGE] PRECISION AND RECALL")
        print(avg_precision)
        print(avg_recall)
        print("SEGMENTATION PERFORMANCE")
        print(mIoU_Class_0)
        print(mIoU_Class_1)
        print(mIoU_Class_2)
        print(mIoU_Class_3)

        print("ALL PIXEL LEVEL [AVERAGE] PRECISION AND RECALL")
        print(avg_all_precision)
        print(avg_all_recall)
        print("PATH LEVEL [AVERAGE] PRECISION AND RECALL")
        print(avg_path_precision)
        print(avg_path_recall)
        print("DURATION RESULTS")
        print(avg_time_net)
        print(avg_time_pp)

    precision_TP_1 = []
    recall_TP_1 = []
    precision_all_1 = []
    recall_all_1 = []
    precision_path_1 = []
    recall_path_1 = []

    if flag_single_multiple_path_evaluation:
        with open(os.path.join(metrics_output_dir, 'precision_TP_1.txt'), 'a') as f:
            for item in res_eval:
                prec = item["precision"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    precision_TP_1.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_TP_1.txt'), 'a') as f:
            for item in res_eval:
                rec = item["recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    recall_TP_1.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'precision_all_1.txt'), 'a') as f:
            for item in res_eval:
                prec = item["all_pixel_prec"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    precision_all_1.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_all_1.txt'), 'a') as f:
            for item in res_eval:
                rec = item["all_pixel_recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    recall_all_1.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'precision_path_1.txt'), 'a') as f:
            for item in res_eval:
                prec = item["path_level_prec"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    precision_path_1.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_path_1.txt'), 'a') as f:
            for item in res_eval:
                rec = item["path_level_recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths == 1:
                    recall_path_1.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")

        precision_TP_multi = []
        recall_TP_multi = []
        precision_all_multi = []
        recall_all_multi = []
        precision_path_multi = []
        recall_path_multi = []

        with open(os.path.join(metrics_output_dir, 'precision_TP_multi.txt'), 'a') as f:
            for item in res_eval:
                prec = item["precision"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    precision_TP_multi.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_TP_multi.txt'), 'a') as f:
            for item in res_eval:
                rec = item["recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    recall_TP_multi.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'precision_all_multi.txt'), 'a') as f:
            for item in res_eval:
                prec = item["all_pixel_prec"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    precision_all_multi.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_all_multi.txt'), 'a') as f:
            for item in res_eval:
                rec = item["all_pixel_recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    recall_all_multi.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'precision_path_multi.txt'), 'a') as f:
            for item in res_eval:
                prec = item["path_level_prec"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    precision_path_multi.append(prec)
                    f.write('%f' % prec)
                    f.write("\n")
        with open(os.path.join(metrics_output_dir, 'recall_path_multi.txt'), 'a') as f:
            for item in res_eval:
                rec = item["path_level_recall"]
                num_GT_paths = item["num_GT_paths"]
                if num_GT_paths > 1:
                    recall_path_multi.append(rec)
                    f.write('%f' % rec)
                    f.write("\n")

        print("######################################")
        print("single-track versus multi-track evaluation")
        print("######################################")

        print("TP SINGLE PRECISION")
        print(np.mean(precision_TP_1))
        print("TP SINGLE RECALL")
        print(np.mean(recall_TP_1))
        print("ALL SINGLE PRECISION")
        print(np.mean(precision_all_1))
        print("ALL SINGLE RECALL")
        print(np.mean(recall_all_1))
        print("PATH SINGLE PRECISION")
        print(np.mean(precision_path_1))
        print("PATH SINGLE RECALL")
        print(np.mean(recall_path_1))
        print("***************************************")
        print("TP MULTI PRECISION")
        print(np.mean(precision_TP_multi))
        print("TP MULTI RECALL")
        print(np.mean(recall_TP_multi))
        print("ALL MULTI PRECISION")
        print(np.mean(precision_all_multi))
        print("ALL MULTI RECALL")
        print(np.mean(recall_all_multi))
        print("PATH MULTI PRECISION")
        print(np.mean(precision_path_multi))
        print("PATH MULTI RECALL")
        print(np.mean(recall_path_multi))
