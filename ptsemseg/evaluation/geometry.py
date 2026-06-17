"""Evaluation geometry and inverse-perspective mapping helpers."""

import math

import cv2
import numpy as np


class MyUtil:
    def __init__(self):
        return

    def rotz(self, theta_in):
        ct = math.cos(theta_in)
        st = math.sin(theta_in)

        mat = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])

        return mat

    def roty(self, theta_in):
        ct = math.cos(theta_in)
        st = math.sin(theta_in)

        mat = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])

        return mat

    def rotx(self, theta_in):
        ct = math.cos(theta_in)
        st = math.sin(theta_in)

        mat = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])

        return mat

    def convert_rot_trans_to_hmat(self, mat_rot, vec_trans):
        hmat = np.zeros(shape=(4, 4))

        hmat[0, 0] = mat_rot[0, 0]
        hmat[0, 1] = mat_rot[0, 1]
        hmat[0, 2] = mat_rot[0, 2]
        hmat[0, 3] = vec_trans[0]

        hmat[1, 0] = mat_rot[1, 0]
        hmat[1, 1] = mat_rot[1, 1]
        hmat[1, 2] = mat_rot[1, 2]
        hmat[1, 3] = vec_trans[1]

        hmat[2, 0] = mat_rot[2, 0]
        hmat[2, 1] = mat_rot[2, 1]
        hmat[2, 2] = mat_rot[2, 2]
        hmat[2, 3] = vec_trans[2]

        hmat[3, 3] = 1.0

        return hmat


class MyIPM:
    """
    Note that we are going through the following transformations:
        @ bev-img -> world
        @ world -> camera
        @ camera -> ori-img
    """

    obj_my_util = MyUtil()
    mat_tf_from_img_ori_to_img_bev = None

    def __init__(
        self,
        mat_K,
        param_scale_pixel_per_meter,
        param_h_img_bev,
        param_w_img_bev,
        param_offset_y,
        param_offset_x,
        theta_rotx_deg,
        vec_trans_cam_in_world,
    ):
        self.mat_K = mat_K
        self.param_scale_pixel_per_meter = param_scale_pixel_per_meter
        self.param_h_img_bev = param_h_img_bev
        self.param_w_img_bev = param_w_img_bev
        self.param_offset_y = param_offset_y
        self.param_offset_x = param_offset_x
        self.theta_rotx_deg = theta_rotx_deg
        self.vec_trans_cam_in_world = vec_trans_cam_in_world

        mat_tf_from_img_bev_to_img_ori = self.get_transform_from_bev_to_img_ori()
        mat_tf_from_img_ori_to_img_bev_ = np.linalg.inv(mat_tf_from_img_bev_to_img_ori)
        self.mat_tf_from_img_ori_to_img_bev = (
            mat_tf_from_img_ori_to_img_bev_
            / mat_tf_from_img_ori_to_img_bev_[2, 2]
        )

    def process(self, img_ori):
        if self.mat_tf_from_img_ori_to_img_bev is None:
            print("No transformation was set.")
            return

        img_bev = cv2.warpPerspective(
            img_ori,
            self.mat_tf_from_img_ori_to_img_bev,
            (self.param_w_img_bev, self.param_h_img_bev),
        )

        return img_bev

    def get_pose_cam_in_world(self):
        theta_rotx_deg = self.theta_rotx_deg
        theta_rotx_rad = theta_rotx_deg * (math.pi / 180.0)

        mat_rot_cam_in_world = self.obj_my_util.rotx(theta_rotx_rad)
        vec_trans_cam_in_world = self.vec_trans_cam_in_world

        hmat_cam_in_world = self.obj_my_util.convert_rot_trans_to_hmat(
            mat_rot_cam_in_world,
            vec_trans_cam_in_world,
        )

        return hmat_cam_in_world

    def get_transform_from_world_to_img_bev(self):
        s = self.param_scale_pixel_per_meter
        offx = self.param_offset_x
        offy = self.param_offset_y

        mat_transform = np.array([[s, 0.0, offx], [0.0, -s, offy], [0.0, 0.0, 1.0]])

        return mat_transform

    def get_transform_from_img_ori_to_bev(self):
        return self.mat_tf_from_img_ori_to_img_bev

    def get_transform_from_bev_to_img_ori(self):
        """
        @ bev-img -> world: by get_transform_from_world_to_img_bev()
        @ world -> camera:  by get_pose_cam_in_world()
        @ camera -> ori-img: by k
        """

        mat_tf_from_cam_to_img_ori = self.mat_K

        hmat_cam_in_world = self.get_pose_cam_in_world()
        hmat_world_in_cam = np.linalg.inv(hmat_cam_in_world)
        mat_tf_from_world_to_cam = hmat_world_in_cam[0:3, [0, 1, 3]]

        mat_tf_from_world_to_img_bev = self.get_transform_from_world_to_img_bev()
        mat_tf_from_img_bev_to_world = np.linalg.inv(mat_tf_from_world_to_img_bev)

        mat_tf_from_img_bev_to_img_ori = (
            mat_tf_from_cam_to_img_ori
            @ mat_tf_from_world_to_cam
            @ mat_tf_from_img_bev_to_world
        )
        mat_tf_from_img_bev_to_img_ori_n = (
            mat_tf_from_img_bev_to_img_ori / mat_tf_from_img_bev_to_img_ori[2, 2]
        )

        return mat_tf_from_img_bev_to_img_ori_n

    def get_transform_from_world_to_img_ori(self):
        """
        @ world -> camera:  by get_pose_cam_in_world()
        @ camera -> ori-img: by k
        """

        mat_tf_from_cam_to_img_ori = self.mat_K

        hmat_cam_in_world = self.get_pose_cam_in_world()
        hmat_world_in_cam = np.linalg.inv(hmat_cam_in_world)
        mat_tf_from_world_to_cam = hmat_world_in_cam[0:3, [0, 1, 3]]

        mat_tf_from_world_to_img_ori = mat_tf_from_cam_to_img_ori @ mat_tf_from_world_to_cam

        return mat_tf_from_world_to_img_ori


class MyUtils_3D:
    m_obj_my_ipm = None
    m_mat_tf_from_img_ori_to_bev = None
    m_mat_tf_from_img_ori_to_world = None
    m_mat_tf_from_world_to_img_bev = None
    m_mat_tf_from_world_to_img_ori = None
    m_param_h_img_bev = None
    m_param_w_img_bev = None

    def __init__(self, dict_args):
        (
            self.m_obj_my_ipm,
            self.m_mat_tf_from_img_ori_to_bev,
            self.m_mat_tf_from_img_ori_to_world,
            self.m_mat_tf_from_world_to_img_bev,
            self.m_mat_tf_from_world_to_img_ori,
            self.m_param_h_img_bev,
            self.m_param_w_img_bev,
        ) = self._init_obj_IPM(dict_args)

    def _init_obj_IPM(self, dict_args):
        param_mat_K = dict_args["param_3D_ipm_camera_intrinsic_matrix"]
        theta_rotx_deg = dict_args["param_3D_ipm_camera_pitch_angle"]
        vec_trans_cam_in_world = dict_args["param_3D_ipm_camera_pos_wrt_ground_plane"]

        param_scale_pixel_per_meter = dict_args["param_3D_ipm_img_pixel_per_meter"]
        param_h_img_bev = dict_args["param_3D_ipm_img_height"]
        param_w_img_bev = dict_args["param_3D_ipm_img_width"]

        param_offset_y = param_h_img_bev - 1
        param_offset_x = param_w_img_bev / 2.0

        obj_my_ipm = MyIPM(
            mat_K=param_mat_K,
            param_scale_pixel_per_meter=param_scale_pixel_per_meter,
            param_h_img_bev=param_h_img_bev,
            param_w_img_bev=param_w_img_bev,
            param_offset_y=param_offset_y,
            param_offset_x=param_offset_x,
            theta_rotx_deg=theta_rotx_deg,
            vec_trans_cam_in_world=vec_trans_cam_in_world,
        )

        mat_tf_from_img_ori_to_bev = obj_my_ipm.get_transform_from_img_ori_to_bev()

        mat_tf_from_world_to_img_ori = obj_my_ipm.get_transform_from_world_to_img_ori()
        mat_tf_from_img_ori_to_world = np.linalg.inv(mat_tf_from_world_to_img_ori)

        mat_tf_from_world_to_img_bev = obj_my_ipm.get_transform_from_world_to_img_bev()

        return (
            obj_my_ipm,
            mat_tf_from_img_ori_to_bev,
            mat_tf_from_img_ori_to_world,
            mat_tf_from_world_to_img_bev,
            mat_tf_from_world_to_img_ori,
            param_h_img_bev,
            param_w_img_bev,
        )

    def get_size_img_bev(self):
        return self.m_param_h_img_bev, self.m_param_w_img_bev

    def create_img_IPM(self, img_in):
        img_ipm_out = self.m_obj_my_ipm.process(img_in)

        if 0:
            cv2.imshow("img_ipm_out", img_ipm_out)
            cv2.waitKey(0)

        return img_ipm_out

    def convert_pnt_img_ori_to_pnt_world(self, pnt_img_ori):
        pnt_world_ = self.m_mat_tf_from_img_ori_to_world @ pnt_img_ori
        pnt_world = pnt_world_ / pnt_world_[2, 0]

        x_world = float(pnt_world[0, 0])
        y_world = float(pnt_world[1, 0])
        z_world = 0.0

        return x_world, y_world, z_world

    def convert_pnt_img_ori_to_pnt_bev(self, pnt_img_ori):
        pnt_bev_ = self.m_mat_tf_from_img_ori_to_bev @ pnt_img_ori
        pnt_bev = pnt_bev_ / pnt_bev_[2, 0]

        x_bev = pnt_bev[0, 0]
        y_bev = pnt_bev[1, 0]

        return x_bev, y_bev

    def convert_pnt_world_to_pnt_bev(self, pnt_world):
        pnt_bev = self.m_mat_tf_from_world_to_img_bev @ pnt_world

        x_bev = pnt_bev[0, 0]
        y_bev = pnt_bev[1, 0]

        return x_bev, y_bev

    def convert_pnt_world_to_pnt_img_ori(self, pnt_world):
        pnt_img_ori = self.m_mat_tf_from_world_to_img_ori @ pnt_world

        x_img_ori = pnt_img_ori[0, 0] / pnt_img_ori[2, 0]
        y_img_ori = pnt_img_ori[1, 0] / pnt_img_ori[2, 0]

        return x_img_ori, y_img_ori


__all__ = ["MyIPM", "MyUtil", "MyUtils_3D"]
