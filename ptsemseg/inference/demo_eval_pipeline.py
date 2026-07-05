"""Shared runnable demo/eval pipeline."""

from ptsemseg.evaluation import save_demo_eval_metric_outputs
from ptsemseg.inference.demo_eval_execution import initialize_demo_eval_execution
from ptsemseg.inference.demo_eval_execution import process_demo_eval_image
from ptsemseg.inference.demo_eval_runtime import initialize_demo_eval_runtime


def run_demo_eval():
    """Run the legacy TPEnet demo/eval flow."""

    ###=====================================================================================================================
    ### 0. setting
    ###=====================================================================================================================
    runtime_context = initialize_demo_eval_runtime()

    title_testrun_this = runtime_context.title_testrun_this
    fname_pathlabel_gt_in = runtime_context.fname_pathlabel_gt_in
    format_fname_img_in = runtime_context.format_fname_img_in
    format_fname_img_out = runtime_context.format_fname_img_out
    w_img = runtime_context.w_img
    dx_valid_a = runtime_context.dx_valid_a
    dx_valid_b = runtime_context.dx_valid_b
    metrics_output_dir = runtime_context.metrics_output_dir
    output_subdirs = runtime_context.output_subdirs
    obj_helper_GT = runtime_context.obj_helper_GT
    list_pathlabel_gt_in = runtime_context.list_pathlabel_gt_in
    totnum_steps = runtime_context.totnum_steps

    ###==================================================================================================================
    ### 1. set parameters
    ###==================================================================================================================
    architecture = runtime_context.architecture    # 0 for TPE-Net - 1 for DLink-Net34 - 2 for erfnet - 3 for BisenetV2 - 4 for segformer - 5 SegHarDNet

    num_seg_classes = runtime_context.num_seg_classes
    num_channel_reg = runtime_context.num_channel_reg

    seg_in_pp = runtime_context.seg_in_pp
    flag_miou = runtime_context.flag_miou

    flag_save_img = runtime_context.flag_save_img
    flag_save_data = runtime_context.flag_save_data
    flag_single_multiple_path_evaluation = runtime_context.flag_single_multiple_path_evaluation

    data_in_use = runtime_context.data_in_use      # 0 for RailSem19 - 1 for RailSet - 2 for RailDB - 3 for YDHR - 4 for others without GT data

    DATASET_for_use = runtime_context.dataset_for_use
    args_oper = runtime_context.args_oper
    args_alg = runtime_context.args_alg

    ###==================================================================================================================
    ### 2. init
    ###==================================================================================================================
    ###==================================================================================================================
    ### 3. loop
    ###==================================================================================================================
    execution_context = initialize_demo_eval_execution(
        args_oper=args_oper,
        args_alg=args_alg,
        num_seg_classes=num_seg_classes,
        num_channel_reg=num_channel_reg,
        seg_in_pp=seg_in_pp,
        architecture=architecture,
    )

    list_fnames_img = execution_context.list_fnames_img

    res_eval = []
    PathExtractor = execution_context.path_extractor
    for my_idx, fname_img_in in enumerate(list_fnames_img):
        metric_record = process_demo_eval_image(
            my_idx=my_idx,
            fname_img_in=fname_img_in,
            list_fnames_img=list_fnames_img,
            args_oper=args_oper,
            path_extractor=PathExtractor,
            data_in_use=data_in_use,
            num_seg_classes=num_seg_classes,
            flag_miou=flag_miou,
            architecture=architecture,
            list_pathlabel_gt_in=list_pathlabel_gt_in,
            obj_helper_GT=obj_helper_GT,
            output_subdirs=output_subdirs,
            flag_save_img=flag_save_img,
        )
        if metric_record is not None:
            res_eval.append(metric_record)

    ### 3.5.7 save performance metrics computation results
    save_demo_eval_metric_outputs(
        res_eval=res_eval,
        metrics_output_dir=metrics_output_dir,
        flag_save_data=flag_save_data,
        data_in_use=data_in_use,
        flag_single_multiple_path_evaluation=flag_single_multiple_path_evaluation,
    )


__all__ = ["run_demo_eval"]
