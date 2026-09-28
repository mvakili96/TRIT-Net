"""Shared runnable demo/eval pipeline."""

from ptsemseg.evaluation import save_demo_eval_metric_outputs
from ptsemseg.inference.demo_eval_execution import initialize_demo_eval_execution
from ptsemseg.inference.demo_eval_execution import process_demo_eval_image
from ptsemseg.inference.demo_eval_runtime import initialize_demo_eval_runtime


def run_demo_eval():
    """Run the legacy TPEnet demo/eval flow."""

    runtime_context = initialize_demo_eval_runtime()
    execution_context = initialize_demo_eval_execution(
        args_oper=runtime_context.args_oper,
        args_alg=runtime_context.args_alg,
        num_seg_classes=runtime_context.num_seg_classes,
        num_channel_reg=runtime_context.num_channel_reg,
        seg_in_pp=runtime_context.seg_in_pp,
        architecture=runtime_context.architecture,
        use_clustering_post_process=runtime_context.runtime_settings["use_clustering_post_process"],
    )

    res_eval = []
    for my_idx, fname_img_in in enumerate(execution_context.list_fnames_img):
        metric_record = process_demo_eval_image(
            my_idx=my_idx,
            fname_img_in=fname_img_in,
            list_fnames_img=execution_context.list_fnames_img,
            args_oper=runtime_context.args_oper,
            path_extractor=execution_context.path_extractor,
            data_in_use=runtime_context.data_in_use,
            num_seg_classes=runtime_context.num_seg_classes,
            flag_miou=runtime_context.flag_miou,
            architecture=runtime_context.architecture,
            list_pathlabel_gt_in=runtime_context.list_pathlabel_gt_in,
            obj_helper_GT=runtime_context.obj_helper_GT,
            output_subdirs=runtime_context.output_subdirs,
            flag_save_img=runtime_context.flag_save_img,
            save_rail_area_mask=runtime_context.runtime_settings["flag_save_rail_area_mask"],
            output_filename_index_offset=runtime_context.runtime_settings["output_filename_index_offset"],
        )
        if metric_record is not None:
            res_eval.append(metric_record)

    save_demo_eval_metric_outputs(
        res_eval=res_eval,
        metrics_output_dir=runtime_context.metrics_output_dir,
        flag_save_data=runtime_context.flag_save_data,
        data_in_use=runtime_context.data_in_use,
        flag_single_multiple_path_evaluation=runtime_context.flag_single_multiple_path_evaluation,
    )


__all__ = ["run_demo_eval"]
