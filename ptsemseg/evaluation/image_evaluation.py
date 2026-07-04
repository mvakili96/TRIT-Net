"""Per-image demo/eval metric record assembly."""


def evaluate_demo_eval_image(
    gt_final_dict_xs_img_rail_LR,
    list_res_paths,
    final_im,
    image_height,
    image_width,
    architecture,
    my_idx,
    img_idx,
    class_0,
    class_1,
    class_2,
    class_3,
    gt_3,
    dict_res_time,
):
    """Create the annotated evaluation image and legacy metric record."""
    from ptsemseg.evaluation.metrics import eval_object_topology

    num_GT_paths = len(gt_final_dict_xs_img_rail_LR)
    evaluator_topolgy = eval_object_topology(
        gt_final_dict_xs_img_rail_LR,
        list_res_paths,
        image_height=image_height,
        image_width=image_width,
        arch=architecture,
    )

    annotated_im, y_minimum = evaluator_topolgy.annotate_gt(final_im)

    matching_mat, matched_ones = evaluator_topolgy.find_matches(4, y_minimum)

    TP, FP, FN = evaluator_topolgy.performance_metrics_values_TP_level(matching_mat, matched_ones)
    path_level_prec, path_level_recall = evaluator_topolgy.performance_metrics_values_path_level(matched_ones, min_rate=0)
    all_pixel_prec, all_pixel_recall = evaluator_topolgy.performance_metrics_values_all_pixel_level(
        matching_mat,
        matched_ones,
    )

    if TP == 0 or y_minimum == -1:
        print("************************************************************************************")
        print(my_idx)
        print("************************************************************************************")

    image_showing_evaluation_res = evaluator_topolgy.create_final_result_on_annotated_image_V2(
        annotated_im,
        matching_mat,
        matched_ones,
    )

    if (TP + FP) > 0:
        metric_record = {
            "id": img_idx,
            "num_GT_paths": num_GT_paths,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": (TP / (TP + FP)),
            "recall": (TP / (TP + FN)),
            "Class_0": class_0,
            "Class_1": class_1,
            "Class_2": class_2,
            "Class_3": class_3,
            "GT_3": gt_3,
            "path_level_prec": path_level_prec,
            "path_level_recall": path_level_recall,
            "all_pixel_prec": all_pixel_prec,
            "all_pixel_recall": all_pixel_recall,
            "time_net": dict_res_time["dtime_ab"],
            "time_pp": dict_res_time["dtime_bc"],
        }
    else:
        metric_record = {
            "id": img_idx,
            "num_GT_paths": num_GT_paths,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": 0,
            "recall": 0,
            "Class_0": class_0,
            "Class_1": class_1,
            "Class_2": class_2,
            "Class_3": class_3,
            "GT_3": gt_3,
            "path_level_prec": path_level_prec,
            "path_level_recall": path_level_recall,
            "all_pixel_prec": all_pixel_prec,
            "all_pixel_recall": all_pixel_recall,
            "time_net": dict_res_time["dtime_ab"],
            "time_pp": dict_res_time["dtime_bc"],
        }

    return image_showing_evaluation_res, metric_record
