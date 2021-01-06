import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import os

import models.motion_gan as models
from trainer.gan_trainers import *
from utils.matrix_transformer import MatrixTransformer as mt
from utils.plot_script import plot_3d_motion, draw_pose_from_cords
import utils.paramUtil as paramUtil
from utils.utils_ import *
from dataProcessing import dataset
from options.evaluate_options import *

if __name__ == "__main__":
    parser = TestOptions()
    opt = parser.parse()
    joints_num = 0
    input_size = 72
    data = None
    label_dec = None
    dim_category = 31
    enumerator = None
    device = torch.device("cuda:" + str(opt.gpu_id) if opt.gpu_id else "cpu")
    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name)
    opt.model_path = os.path.join(opt.save_root, 'model')
    opt.joints_path = os.path.join(opt.save_root, 'joints')
    model_file_path = os.path.join(opt.model_path, opt.which_epoch + '.tar')
    result_path = os.path.join(opt.result_path, opt.dataset_type, opt.name)

    if opt.dataset_type == "shihao":
        input_size = 72
        joints_num = 24
        label_dec = list(paramUtil.shihao_coarse_action_enumerator.keys())
        dim_category = len(label_dec)
        enumerator = paramUtil.shihao_coarse_action_enumerator
        dataset_path = "./dataset/pose"
        pkl_path = './dataset/pose_shihao_merge'
    elif opt.dataset_type == "ntu_rgbd":
        joints_num = 25
        input_size = 75
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        dim_category = len(paramUtil.ntu_action_labels)
        enumerator = paramUtil.ntu_action_enumerator
    elif opt.dataset_type == "ntu_rgbd_v2":
        joints_num = 19
        input_size = 57
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        dim_category = len(paramUtil.ntu_action_labels)
        enumerator = paramUtil.ntu_action_enumerator
    elif opt.dataset_type == "ntu_rgbd_vibe":
        joints_num = 18
        input_size = 54
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        dim_category = len(paramUtil.ntu_action_labels)
        enumerator = paramUtil.ntu_action_enumerator

    model = torch.load(model_file_path)
    if opt.use_lie:
        if opt.no_trajectory:
            output_size = input_size - 3
        else:
            output_size = input_size
        generator = models.MotionGeneratorLie(opt.dim_z, dim_category, opt.motion_length, opt.hidden_size, opt,
                                           input_size=input_size, output_size=output_size).to(device)
    else:
        generator = models.MotionGenerator(opt.dim_z, dim_category, opt.motion_length, opt.hidden_size, opt,
                                           input_size=input_size, output_size=input_size).to(device)
    generator.load_state_dict(model['generator'])

    categories = np.arange(dim_category).repeat(opt.replic_times, axis=0)
    num_samples = categories.shape[0]
    category_oh, classes = generator.fix_z_categ(categories)
    fixed_m_noise = generator.sample_z_r(num_samples)
    if opt.use_lie:
        data = dataset.MotionFolderDatasetShihaoV2(opt.clip_set, dataset_path, pkl_path, opt, lie_enforce=False)
        pose_dataset = dataset.PoseDataset(data, opt.lie_enforce, opt.no_trajectory)
        pose_loader = DataLoader(pose_dataset, batch_size=opt.pose_batch, drop_last=True, num_workers=2, shuffle=True)
        motion_loader = None
        if opt.lie_enforce:
            trainer = TrainerLieV2(pose_loader, motion_loader, opt, device, paramUtil.shihao_raw_offsets,
                                 paramUtil.shihao_kinematic_chain)
        else:
            trainer = TrainerLie(pose_loader, motion_loader, opt, device, paramUtil.shihao_raw_offsets,
                                 paramUtil.shihao_kinematic_chain)
        fake_motion = trainer.evaluate(generator, fixed_m_noise, category_oh, num_samples).numpy()
    else:
        fake_motion = Trainer.evaluate(generator, fixed_m_noise, category_oh, num_samples).numpy()

    print(fake_motion.shape)
    # print(fake_motion[:, 0, :2])
    for i in range(fake_motion.shape[0]):
        class_type = enumerator[label_dec[classes[i]]]
        motion_orig = fake_motion[i]
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        keypoint_path = os.path.join(result_path, 'keypoint')
        if not os.path.exists(keypoint_path):
            os.makedirs(keypoint_path)
        file_name = os.path.join(result_path, class_type + str(i) + ".gif")
        offset = np.matlib.repmat(np.array([motion_orig[0, 0], motion_orig[0, 1], motion_orig[0, 2]]),
                                     motion_orig.shape[0], joints_num)
        # offset = np.tile(motion_orig[:, :3], 24)
        # print(offset[1])
        motion_mat = motion_orig - offset

        motion_mat = motion_mat.reshape(-1, joints_num, 3)
        # motion_mat[:, :, 2] *= -1

        np.save(os.path.join(keypoint_path, class_type + str(i) + '.npy'), motion_mat)
        if opt.dataset_type == "shihao":
            pose_tree = paramUtil.smpl_tree

            # offset = np.tile(np.array([-0.43391575,  0.31606525,  2.57938163]), (motion_orig.shape[0], 24, 1))
            # motion_3d = offset + motion_mat
            # motion_3d = motion_3d.reshape(-1, joints_num, 3)
            # print(motion_3d[..., 2].min(), motion_3d[..., 2].max())
            # motion_2d_mat = np.zeros(motion_3d.shape[:-1] + (2,))
            # motion_2d_imgs = []
            # for k in range(motion_3d.shape[0]):
            #     motion_2d_mat[k] = project_3d_to_2d(motion_3d[k])
            #
            # crop_bbox, crop_size = compute_videocrop_bbox(motion_2d_mat, 100, 1.28, (1920, 1080), thresold=0)
            # motion_2d_mat = crop_and_resize_motion(motion_2d_mat, crop_bbox, crop_size, (256, 200), joints_num=24)
            # for k in range(motion_2d_mat.shape[0]):
            #     img_2d = draw_pose_from_cords((200, 256), motion_2d_mat[k], pose_tree, 2)
            #     motion_2d_imgs.append(img_2d)


            # file_prefix = result_path + class_type
            # compose_gif_img_list(motion_2d_imgs, file_prefix + '_2d.gif', duration=0.5)
            # np.save(os.path.join(keypoint_path, class_type + '_3d.npy'), motion_3d)
            # np.save(os.path.join(keypoint_path, class_type + '_2d.npy'), motion_2d_mat)

            motion_mat = mt.swap_yz(motion_mat)
            motion_mat[:, :, 2] *= -1
            plot_3d_motion(motion_mat, pose_tree, class_type, file_name, interval=150)

        elif opt.dataset_type == "ntu_rgbd":
            motion_mat = mt.swap_xz(motion_mat)
            pose_tree = paramUtil.kinect_tree_exclude
            exclued_points = paramUtil.excluded_joint_ids
            plot_3d_motion(motion_mat, pose_tree, class_type, file_name, interval=150,
                           excluded_joints=exclued_points)
        elif opt.dataset_type == "ntu_rgbd_v2":
            motion_mat = mt.swap_yz(motion_mat)
            pose_tree = paramUtil.kinect_tree_v2
            # exclued_points = paramUtil.excluded_joint_ids
            plot_3d_motion(motion_mat, pose_tree, class_type, file_name, interval=150)
        elif opt.dataset_type == "ntu_rgbd_vibe":
            motion_mat = mt.swap_yz(motion_mat)
            motion_mat[:, :, 2] = -1 * motion_mat[:, :, 2]
            motion_mat = mt.rotate_along_z(motion_mat, 180)
            pose_tree = paramUtil.kinect_tree_vibe
            # exclued_points = paramUtil.excluded_joint_ids
            plot_3d_motion(motion_mat, pose_tree, class_type, file_name, interval=150)
