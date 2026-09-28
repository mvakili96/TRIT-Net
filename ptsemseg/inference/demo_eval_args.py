import argparse
import pickle

import numpy as np

from ptsemseg.inference.runtime_defaults import get_algorithm_runtime_defaults
from ptsemseg.inference.runtime_defaults import get_camera_calibration_path
from ptsemseg.inference.runtime_defaults import get_operation_runtime_defaults


def _get_runtime_default_helpers():
    return (
        get_camera_calibration_path,
        get_algorithm_runtime_defaults,
        get_operation_runtime_defaults,
    )


########################################################################################################################
###
########################################################################################################################
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')
    #end
#END


########################################################################################################################
### define parameters for operation (outside PE_TPEnet)
########################################################################################################################
def define_args_operation(data_in_use, architecture):
    _, _, get_operation_runtime_defaults = _get_runtime_default_helpers()
    operation_defaults = get_operation_runtime_defaults(data_in_use, architecture)

    parser = argparse.ArgumentParser(description="Params_operation")
    parser.add_argument(
        '--size_img_process',
        type=int,
        default=operation_defaults["size_img_process"],
        help='image size used in a process',
    )
    parser.set_defaults(dir_input=operation_defaults["dir_input"])
    parser.add_argument('--dir_output', type=str, default=operation_defaults["dir_output"], help='directory for output images')
    parser.add_argument(
        '--b_save_res_imgs_as_file',
        type=str2bool,
        nargs='?',
        const=True,
        default=operation_defaults["b_save_res_imgs_as_file"],
        help='Save res imgs as file?',
    )

    return parser
#END



########################################################################################################################
### define parameters for algorithm (inside PE_TPEnet)
########################################################################################################################
def define_args_algorithm():
    parser = argparse.ArgumentParser(description="Params_algorithm")
    _, get_algorithm_runtime_defaults, _ = _get_runtime_default_helpers()
    algorithm_defaults = get_algorithm_runtime_defaults()


    ###===================================================================================================
    ### For YDHR
    ###===================================================================================================

    ### basic
    parser.add_argument(
        '--b_create_imgs_res_interim',
        type=str2bool,
        nargs='?',
        const=True,
        default=algorithm_defaults["b_create_imgs_res_interim"],
        help='create interim res images',
    )

    ### network weight
    # TPEnet_a_weights_20800
    # Mybest_7000_30000_only_cen
    # Mybest_7000_39000_2_only_cen_noseg          Mybest_7000_83700_2_only_cen
    parser.add_argument(
        '--file_weight',
        type=str,
        default=algorithm_defaults["file_weight"],
        help='location of network weight file',
    )

    ### params for detection of local maxima triplets
    parser.add_argument('--param_triplet_nms_alpha', type=float, default=algorithm_defaults["param_triplet_nms_alpha"], help='param (alpha) for detection of local maxima triplets')
    parser.add_argument('--param_triplet_nms_beta', type=float, default=algorithm_defaults["param_triplet_nms_beta"], help='param (beta) for detection of local maxima triplets')
    parser.add_argument('--param_triplet_nms_min', type=float, default=algorithm_defaults["param_triplet_nms_min"], help='param (min) for detection of local maxima triplets')
    parser.add_argument('--param_triplet_nms_scale', type=float, default=algorithm_defaults["param_triplet_nms_scale"], help='param (scale) for detection of local maxima triplets')


    ### params for inverse perspective mapping (for 3d data) -> would be re-set by set_value_for_args_algorithm()
    parser.add_argument('--param_3D_ipm_camera_intrinsic_matrix', type=float, default=np.zeros(shape=(3,3)), help='param for 3D data - camera intrinsic matrix')
    parser.add_argument('--param_3D_ipm_camera_pitch_angle', type=float, default=algorithm_defaults["param_3D_ipm_camera_pitch_angle"], help='param for 3D data - camera pitch angle (deg)')
    parser.add_argument('--param_3D_ipm_camera_pos_wrt_ground_plane', type=float, default=np.array(algorithm_defaults["param_3D_ipm_camera_pos_wrt_ground_plane"]), help='param for 3D data - camera position wrt ground plane (meter)')
    parser.add_argument('--param_3D_ipm_img_pixel_per_meter', type=float, default=algorithm_defaults["param_3D_ipm_img_pixel_per_meter"], help='pixel per meter in ipm-img')
    parser.add_argument('--param_3D_ipm_img_height', type=int, default=algorithm_defaults["param_3D_ipm_img_height"], help='height of ipm-img')
    parser.add_argument('--param_3D_ipm_img_width', type=int, default=algorithm_defaults["param_3D_ipm_img_width"], help='width of ipm-img')

    ### params for rail path graph
    parser.add_argument('--param_rpg_subedge_thres_dx_3d', type=float, default=algorithm_defaults["param_rpg_subedge_thres_dx_3d"], help='(meter)')
    parser.add_argument('--param_rpg_subedge_thres_dy_img', type=int, default=algorithm_defaults["param_rpg_subedge_thres_dy_img"], help='(pixel)')
    parser.add_argument('--param_rpg_subedge_height_section', type=int, default=algorithm_defaults["param_rpg_subedge_height_section"], help='(pixel)')

    parser.add_argument('--param_rpg_nodeedge_thres_dist_img_for_seed', type=int, default=algorithm_defaults["param_rpg_nodeedge_thres_dist_img_for_seed"], help='(pixel)')
    parser.add_argument('--param_rpg_nodeedge_thres_dx_3d', type=float, default=algorithm_defaults["param_rpg_nodeedge_thres_dx_3d"], help='(meter)')
    parser.add_argument('--param_rpg_nodeedge_thres_dy_img', type=int, default=algorithm_defaults["param_rpg_nodeedge_thres_dy_img"], help='(pixel)')

    parser.add_argument('--param_rpg_path_vertices_valid_y_min', type=float, default=algorithm_defaults["param_rpg_path_vertices_valid_y_min"], help='(meter)')

    parser.add_argument('--param_rpg_poly_fitting_y_max', type=float, default=algorithm_defaults["param_rpg_poly_fitting_y_max"], help='(meter)')
    parser.add_argument('--param_rpg_poly_fitting_degree', type=int, default=algorithm_defaults["param_rpg_poly_fitting_degree"], help='degree of polynomial')



    return parser
#END


########################################################################################################################
### define parameters for algorithm (inside PE_TPEnet)
########################################################################################################################
def set_value_for_args_algorithm(args):
    """
    In this function, we set values for the following args:
        args.param_3D_ipm_camera_intrinsic_matrix
        args.param_3D_ipm_camera_pitch_angle
        args.param_3D_ipm_camera_pos_wrt_ground_plane
        args.param_3D_ipm_img_pixel_per_meter
        args.param_3D_ipm_img_height
        args.param_3D_ipm_img_width

    :param args:
    :return:
    """

    get_camera_calibration_path, get_algorithm_runtime_defaults, _ = _get_runtime_default_helpers()
    algorithm_defaults = get_algorithm_runtime_defaults()

    ###===================================================================================================
    ### For YDHR
    ###===================================================================================================

    ###---------------------------------------------------------------------------------------------------
    ### read camera intrinsic matrix from file
    ###---------------------------------------------------------------------------------------------------
    file_camera_calib = get_camera_calibration_path()
        # The file './camera_calib/params_cam_calib_960_540.pickle' is
        #   obtained using gopro images used in YDHR Oct 9&10 data collection.

    dict_temp = None

    with open(file_camera_calib, 'rb') as fh:
        dict_temp = pickle.load(fh)
    # end

    param_mat_K = dict_temp['k_mat']  # (3x3) ndarr

    ###---------------------------------------------------------------------------------------------------
    ### set value for args
    ###---------------------------------------------------------------------------------------------------
    args.param_3D_ipm_camera_intrinsic_matrix = param_mat_K
    args.param_3D_ipm_camera_pitch_angle = algorithm_defaults["param_3D_ipm_camera_pitch_angle"]
    args.param_3D_ipm_camera_pos_wrt_ground_plane = np.array(algorithm_defaults["param_3D_ipm_camera_pos_wrt_ground_plane"])
    args.param_3D_ipm_img_pixel_per_meter = algorithm_defaults["param_3D_ipm_img_pixel_per_meter"]
    args.param_3D_ipm_img_height = algorithm_defaults["param_3D_ipm_img_height"]
    args.param_3D_ipm_img_width = algorithm_defaults["param_3D_ipm_img_width"]


    return args
#END


__all__ = [
    "define_args_algorithm",
    "define_args_operation",
    "set_value_for_args_algorithm",
    "str2bool",
]
