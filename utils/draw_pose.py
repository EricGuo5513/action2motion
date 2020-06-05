import csv
import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import dataProcessing.dataset
import utils.paramUtil as paramUtil
import os
from utils.matrix_transformer import MatrixTransformer as mt
from utils.plot_script import plot_3d_motion, plot_2d_motion, draw_pose_from_cords, plot_2d_pose
from utils.utils_ import compose_gif_img_list, project_3d_to_2d

filename = "../dataset/pose/chenxiangye_group1_time1/pose.csv"

with open(filename, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data).astype(float)


motion_orig = data[:20, 1:]

# for generated data from shihao's dataset(motion_gan)
'''

'''
# for generated data from shihao's dataset(encoder_decoder)
'''
generated_motion = np.load("../3d_joints/endecoder/201.0.npy").reshape((1, -1, 72))
save_file_prefix = '../3d_joints/'
'''
# for generated data from ntu's dataset(motion_gan)

generated_motion = np.load("../checkpoints/vae/vae_test/joints/motion_joints1900.npy")
classes = np.load("../checkpoints/vae/vae_test/joints/motion_class.npy")
save_file_prefix = '../checkpoints/vae/vae_test/joints/1900/'
label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]

if not os.path.exists(save_file_prefix):
    os.makedirs(save_file_prefix)

def animate_ntu_vibe(generated_motion):
    enumerator = paramUtil.ntu_action_enumerator
    for i in range(generated_motion.shape[0]):
        class_type = enumerator[label_dec[classes[i]]]
        data = generated_motion[i]
        offset = numpy.matlib.repmat(np.array([data[0, 0], data[0, 1], data[0, 2]]), data.shape[0], 18)
        motion_mat = data - offset
        motion_mat = motion_mat.reshape(-1, 18, 3)
        # print(motion_mat[0][1])
        # motion_mat = mt.swap_xz(motion_mat)
        # print(motion_mat[0][1])
        # motion_mat = mt.swap_xy(motion_mat)
        # print(motion_mat[0][1])
        motion_mat = mt.swap_yz(motion_mat)
        motion_mat[:, :, 2] = -1 * motion_mat[:, :, 2]
        motion_mat = mt.rotate_along_z(motion_mat, 180)
        save_path_3d = save_file_prefix + class_type + "_" + str(i) + ".gif"
        pose_tree = paramUtil.kinect_tree_vibe
        # exclued_points = paramUtil.excluded_joint_ids
        plot_3d_motion(motion_mat, pose_tree, class_type, save_path_3d, interval=150)


def animate_ntu_v2(generated_motion):
    enumerator = paramUtil.ntu_action_enumerator
    for i in range(generated_motion.shape[0]):
        class_type = enumerator[label_dec[classes[i]]]
        data = generated_motion[i]
        offset = numpy.matlib.repmat(np.array([data[0, 0], data[0, 1], data[0, 2]]), data.shape[0], 19)
        motion_mat = data - offset
        motion_mat = motion_mat.reshape(-1, 19, 3)
        # print(motion_mat[0][1])
        # motion_mat = mt.swap_xz(motion_mat)
        # print(motion_mat[0][1])
        # motion_mat = mt.swap_xy(motion_mat)
        # print(motion_mat[0][1])
        motion_mat = mt.swap_yz(motion_mat)
        save_path_3d = save_file_prefix + class_type + "_" + str(i) + ".gif"
        pose_tree = paramUtil.kinect_tree_v2
        # exclued_points = paramUtil.excluded_joint_ids
        try:
            plot_3d_motion(motion_mat, pose_tree, class_type, save_path_3d, interval=150)
        except Exception:
            continue


def animate_ntu(generated_motion):
    enumerator = paramUtil.ntu_action_enumerator
    for i in range(generated_motion.shape[0]):
        class_type = enumerator[label_dec[classes[i]]]
        data = generated_motion[i]
        offset = numpy.matlib.repmat(np.array([data[0, 0], data[0, 1], data[0, 2]]), data.shape[0], 25)
        motion_mat = data - offset
        motion_mat = motion_mat.reshape(-1, 25, 3)
        motion_mat = mt.swap_xz(motion_mat)
        save_path_3d = save_file_prefix + class_type + "_" + str(i) + ".gif"
        pose_tree = paramUtil.kinect_tree_exclude
        exclued_points = paramUtil.excluded_joint_ids
        try:
            plot_3d_motion(motion_mat, pose_tree, class_type, save_path_3d, interval=150, excluded_joints=exclued_points)
        except Exception:
            continue


def animate_shihao(generated_motion):
    # for motion_gan
    generated_motion = np.swapaxes(generated_motion, 0, 1)
    # print(generated_motion.shape)
    # real_motion = MotionFolderDataset("../dataset/pose_clip.csv", "../dataset/pose")
    enumerator = paramUtil.shihao_action_enumerator

    for i in range(generated_motion.shape[0]):
        class_type = enumerator[classes[i]]
        # class_type = 'walk'
        motion_mat = generated_motion[i]
    # for i in range(real_motion.__len__()):
    #     motion_orig, label = real_motion.__getitem__(i)
    #     class_type = enumerator[label]
        file_name = save_file_prefix + class_type + "_" + str(i) + ".gif"

        motion_mat = motion_mat.reshape(-1, 24, 3)
        pose_tree = paramUtil.smpl_tree

        motion_2d_mat = np.zeros(motion_mat.shape[:-1] + (2,))
        motion_2d_imgs = []
        motion_orig = motion_mat + np.tile(np.array([-0.43391575,  0.31606525,  2.57938163]), (motion_mat.shape[0], 24, 1))
        for k in range(motion_orig.shape[0]):
            motion_2d_mat[k] = project_3d_to_2d(motion_orig[k])
            img_2d = draw_pose_from_cords((1080, 1920), motion_2d_mat[k], pose_tree, 2)
            motion_2d_imgs.append(img_2d)
        # print(motion_2d_mat[0])

        file_prefix = save_file_prefix + class_type + "_" + str(i)
        compose_gif_img_list(motion_2d_imgs, file_prefix + '_2d.gif', duration=0.5)
        np.save(file_prefix + '_2d.npy', motion_2d_mat)

        motion_mat = motion_mat.reshape(-1, 24, 3)
        # motion_mat[:, :, 2] *= -1
        motion_mat = mt.swap_yz(motion_mat)
        motion_mat[:, :, 2] *= -1
        # motion_mat[:, :, 1] *= -1

        # print(motion_mat[2, 0])
        plot_3d_motion(motion_mat, paramUtil.smpl_tree, class_type, file_name, interval=150)

# animate_shihao(generated_motion)
animate_ntu_vibe(generated_motion)


pose_3d = np.array([[-0.24014706,  0.26219216,  2.58665342],
        [-0.15573644,  0.33471124,  2.62324985],
        [-0.31203248,  0.35072854,  2.57955996],
        [-0.25528682,  0.16766486,  2.59776028],
        [-0.1270656 ,  0.68639939,  2.71411186],
        [-0.34430743,  0.71663586,  2.64085508],
        [-0.25818166,  0.07172318,  2.57813017],
        [-0.12416402,  1.06108656,  2.80831846],
        [-0.32219954,  1.0861528 ,  2.76528242],
        [-0.25131804,  0.02462067,  2.55782521],
        [-0.07859268,  1.17188331,  2.75248727],
        [-0.35937683,  1.16403054,  2.66327208],
        [-0.2590769 , -0.2060026 ,  2.58304342],
        [-0.17436036, -0.1040472 ,  2.60991678],
        [-0.35035909, -0.10013865,  2.5646342 ],
        [-0.24011179, -0.26109949,  2.50871335],
        [-0.06964335, -0.13927979,  2.62266431],
        [-0.44485904, -0.1457554 ,  2.51163239],
        [ 0.01724727,  0.11438849,  2.61446151],
        [-0.52816403,  0.10183121,  2.43378336],
        [ 0.05920819, -0.11941175,  2.56809198],
        [-0.51157453, -0.1502803 ,  2.39274296],
        [ 0.06497607, -0.19859481,  2.55630928],
        [-0.4761446 , -0.22781689,  2.39491161]])
# print(project_3d_to_2d(pose_3d))