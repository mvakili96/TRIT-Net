"""Demo/eval result-image writers.

The filenames intentionally preserve the legacy ``resluting_image_*`` typo and
the existing IMG/SEG/CEN/AFM output split.
"""

import os

import cv2


def save_demo_eval_result_images(
    output_subdirs,
    img_idx,
    image_showing_evaluation_res,
    img_res_seg,
    img_res_centerness,
    img_res_AFM_direct,
):
    """Write legacy demo/eval result images."""

    cv2.imwrite(os.path.join(output_subdirs["img"], "resluting_image_" + str(img_idx) + ".jpg"), image_showing_evaluation_res)
    cv2.imwrite(os.path.join(output_subdirs["seg"], "resluting_image_" + str(img_idx) + ".bmp"), img_res_seg)
    cv2.imwrite(os.path.join(output_subdirs["cen"], "resluting_image_" + str(img_idx) + ".png"), img_res_centerness)
    if img_res_AFM_direct is not None:
        cv2.imwrite(os.path.join(output_subdirs["afm"], "resluting_image_" + str(img_idx) + ".png"), img_res_AFM_direct)
