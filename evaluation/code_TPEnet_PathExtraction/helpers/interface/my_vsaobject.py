"""Compatibility wrapper for legacy demo/eval VSA object helpers."""

from ptsemseg.evaluation.vsa import Polygon_dummy
from ptsemseg.evaluation.vsa import create_VSAObject_from_PE_results

__all__ = [
    "Polygon_dummy",
    "create_VSAObject_from_PE_results",
]

    #---------------------------------------------------------------------------------------------------
    # list_paths_out:
    #   list_paths_out[i]: ith path, is {dict:3}
    #       'extracted': having the following
    #            -> set in def _convert_to_paths_as_vertices_v2(..):
    #            dict_path_this = {"id_edge": [],
    #                              "xy_cen_img": [],
    #                              "xy_left_img": [],
    #                              "xy_right_img": [],
    #                              "xyz_cen_3d": [],
    #                              "xyz_left_3d": [],
    #                              "xyz_right_3d": [],
    #                              ###
    #                              "id_node_switch": [],  # switch (id_node)
    #                              "xy_switch_img": [],  # switch (img)
    #                              "xyz_switch_3d": [],  # switch (3d)
    #                              ###
    #                              "xy_switch_img_edge_start": [],
    #                              "xyz_switch_3d_edge_start": [],
    #                              "xy_switch_img_edge_end": [],  # equal to "xy_switch_img"
    #                              "xyz_switch_3d_edge_end": []  # equal to "xyz_switch_3d"
    #                              }
    #
    #       'polynomial': having the following dict
    #           -> set in def _get_paths_by_polynomial_fitting(..):
    #            dict_path_poly_this = {"xyz_cen_3d": sample_arr_xyz_cen_ori,
    #                                   "xyz_left_3d": sample_arr_xyz_left_ori,
    #                                   "xyz_right_3d": sample_arr_xyz_right_ori,
    #                                   "coeff_poly_cen_3d_new": coeff_poly_cen,
    #                                   "coeff_poly_left_3d_new": coeff_poly_left,
    #                                   "coeff_poly_right_3d_new": coeff_poly_right}
    #
    #       'type_path': list_type_paths[idx_path] -> EGO or NON-EGO
    #
    #---------------------------------------------------------------------------------------------------

