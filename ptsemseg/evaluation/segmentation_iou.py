"""Segmentation IoU helpers for demo/eval."""

import cv2


def calculate_demo_eval_segmentation_iou(
    data_in_use,
    flag_miou,
    gt_segmentation,
    labels_seg_predicted,
    image_height,
    image_width,
    num_seg_classes,
    class_0,
    class_1,
    class_2,
    class_3,
    gt_3,
):
    """Calculate legacy demo/eval segmentation IoU fields."""

    if data_in_use <= 2:
        if flag_miou:
            from ptsemseg.evaluation.metrics import eval_seg_object

            gt_segmentation = cv2.resize(gt_segmentation, (image_width, image_height))

            evaluator_seg = eval_seg_object(
                gt_segmentation,
                labels_seg_predicted,
                image_height=image_height,
                image_width=image_width,
            )

            if num_seg_classes == 3:
                _, class_0 = evaluator_seg.calculate_IoU(class_this=0)
                _, class_1 = evaluator_seg.calculate_IoU(class_this=1)
                _, class_2 = evaluator_seg.calculate_IoU(class_this=2)
                gt_3 = 0.1

            if num_seg_classes == 4:
                _, class_0 = evaluator_seg.calculate_IoU(class_this=0)
                _, class_1 = evaluator_seg.calculate_IoU(class_this=1)
                _, class_2 = evaluator_seg.calculate_IoU(class_this=2)
                gt_3, class_3 = evaluator_seg.calculate_IoU(class_this=3)

            if num_seg_classes == 19:
                _, class_0 = evaluator_seg.calculate_IoU(class_this=12)
                _, class_1 = evaluator_seg.calculate_IoU(class_this=17)
                _, class_2 = evaluator_seg.calculate_IoU(class_this=3)

        else:
            gt_3 = 0.1

    return {
        "gt_segmentation": gt_segmentation,
        "Class_0": class_0,
        "Class_1": class_1,
        "Class_2": class_2,
        "Class_3": class_3,
        "GT_3": gt_3,
    }
