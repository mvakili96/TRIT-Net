"""VSA object helpers for path-extraction evaluation results."""

from ptsemseg.evaluation.types import TYPE_path


class Polygon_dummy:
    def __init__(self, vertices=None, coefficients=None):
        self.vertices = vertices
        self.coefficients = coefficients


def create_VSAObject_from_PE_results(list_res_paths):
    detections = []

    for list_path_this in list_res_paths:
        type_path = list_path_this["type_path"]

        if type_path is TYPE_path.EGO:
            xyz_left_3d = list_path_this["polynomial"]["xyz_left_3d"]
            xyz_right_3d = list_path_this["polynomial"]["xyz_right_3d"]

            coeff_poly_left_3d_new = list_path_this["polynomial"]["coeff_poly_left_3d_new"]
            coeff_poly_right_3d_new = list_path_this["polynomial"]["coeff_poly_right_3d_new"]

            detections.append(Polygon_dummy(vertices=xyz_left_3d, coefficients=coeff_poly_left_3d_new))
            detections.append(Polygon_dummy(vertices=xyz_right_3d, coefficients=coeff_poly_right_3d_new))

    return detections
