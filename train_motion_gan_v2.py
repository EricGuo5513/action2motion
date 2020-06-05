"""
Usage:
    train_motion_gan.py [options] <dataset>
Options:
    --image_dataset=<path>          specifies a separate dataset to train for images [default: ]
    --image_batch=<count>           number of images in image batch [default: 10]
    --video_batch=<count>           number of videos in video batch [default: 3]
    --image_size=<int>              resize all frames to this size [default: 64]
    --use_infogan                   when specified infogan loss is used
    --use_categories                when specified ground truth categories are used to
                                    train CategoricalVideoDiscriminator
    --use_noise                     when specified instance noise is used
    --noise_sigma=<float>           when use_noise is specified, noise_sigma controls
                                    the magnitude of the noise [default: 0]
    --image_discriminator=<type>    specifies image disciminator type (see models.py for a
                                    list of available models) [default: PatchImageDiscriminator]
    --video_discriminator=<type>    specifies video discriminator type (see models.py for a
                                    list of available models) [default: CategoricalVideoDiscriminator]
    --video_length=<len>            length of the video [default: 16]
    --print_every=<count>           print every iterations [default: 1]
    --n_channels=<count>            number of channels in the input data [default: 3]
    --every_nth=<count>             sample training videos using every nth frame [default: 4]
    --batches=<count>               specify number of batches to train [default: 100000]
    --dim_z_content=<count>         dimensionality of the content input, ie hidden space [default: 50]
    --dim_z_motion=<count>          dimensionality of the motion input [default: 10]
    --dim_z_category=<count>        dimensionality of categorical input [default: 6]
"""
import argparse
import torch
from torch.utils.data import DataLoader

import models.motion_gan as models
import utils.paramUtil as paramUtil
from trainer.gan_trainers import *
from dataProcessing import dataset
from utils.plot_script import plot_loss
from options.train_options import TrainOptions
import os


if __name__ == "__main__":
    parser = TrainOptions()
    opt = parser.parse()
    device = torch.device("cuda:" + str(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    opt.save_root = os.path.join(opt.checkpoints_dir, opt.name)
    opt.model_path = os.path.join(opt.save_root, 'model')
    opt.joints_path = os.path.join(opt.save_root, 'joints')
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.exists(opt.joints_path):
        os.makedirs(opt.joints_path)

    dataset_path = ""
    joints_num = 0
    input_size = 72
    data = None

    if opt.dataset_type == "shihao":
        dataset_path = "./dataset/pose"
        input_size = 72
        joints_num = 24
        data = dataset.MotionFolderDatasetShihao(opt.clip_set, dataset_path)
    elif opt.dataset_type == "ntu_rgbd":
        file_prefix = "./dataset/"
        motion_desc_file = "motionlist.txt"
        joints_num = 25
        input_size = 75
        labels = paramUtil.ntu_action_labels
        data = dataset.MotionFolderDatasetNTU(file_prefix, motion_desc_file, labels, offset=True, exclude_joints=paramUtil.excluded_joint_ids)
    elif opt.dataset_type == "ntu_rgbd_v2":
        file_prefix = "./dataset/"
        motion_desc_file = "motionlistv2.txt"
        joints_num = 19
        input_size = 57
        labels = paramUtil.ntu_action_labels
        data = dataset.MotionFolderDatasetNTU(file_prefix, motion_desc_file, labels, joints_num=joints_num,
                                              offset=True)
    elif opt.dataset_type == "ntu_rgbd_vibe":
        file_prefix = "./dataset"
        motion_desc_file = "ntu_vibe_list.txt"
        joints_num = 18
        input_size = 54
        labels = paramUtil.ntu_action_labels
        data = dataset.MotionFolderDatasetNtuVIBE(file_prefix, motion_desc_file, labels, joints_num=joints_num,
                                              offset=True, extract_joints=paramUtil.kinect_vibe_extract_joints)
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    dim_category = len(data.labels)
    pose_dataset = dataset.PoseDataset(data)
    pose_loader = DataLoader(pose_dataset, batch_size=opt.pose_batch, drop_last=True, num_workers=2, shuffle=True)

    motion_dataset = dataset.MotionDataset(data, opt.motion_length)
    motion_loader = DataLoader(motion_dataset, batch_size=opt.motion_batch, drop_last=True, num_workers=2, shuffle=True)

    generator = models.MotionGenerator(opt.dim_z, dim_category, opt.motion_length, opt.hidden_size, joints_num=joints_num
                                       , input_size=input_size, output_size=input_size)

    if opt.lie_enforce and opt.no_trajectory:
        input_size = input_size - 3
    motion_discriminator = models.MotionDiscriminator(input_size, opt.hidden_size, opt.hidden_layer, output_size=1)
    motion_classifier = models.MotionDiscriminator(input_size, opt.hidden_size, opt.hidden_layer, output_size=dim_category)
    pose_discriminator = models.PoseDiscriminator(input_size, opt.hidden_size)

    pc_g = sum(param.numel() for param in generator.parameters())
    pc_pose_dis = sum(param.numel() for param in pose_discriminator.parameters())
    pc_motion_dis = sum(param.numel() for param in motion_discriminator.parameters())
    pc_motion_cls = sum(param.numel() for param in motion_classifier.parameters())
    print("Total parameters of generator: {}".format(pc_g))
    print("Total parameters of pose discriminator: {}".format(pc_pose_dis))
    print("Total parameters of motion discriminator: {}".format(pc_motion_dis))
    print("Total parameters of motion discriminator: {}".format(pc_motion_cls))
    print("Total parameters of Motion GAN: {}".format(pc_g + pc_pose_dis + pc_motion_dis + pc_motion_dis))

    trainer = TrainerV2(pose_loader, motion_loader, opt, device)

    logs = trainer.train_v2(generator, pose_discriminator, motion_discriminator, motion_classifier)

    plot_loss(logs, opt.save_root + "loss_curve.png", opt.plot_every)

    state = {
        "generator": generator.state_dict(),
        "pose_discriminator": pose_discriminator.state_dict(),
        "motion_discriminator": motion_discriminator.state_dict(),
        "opt_generator": trainer.opt_generator.state_dict(),
        "opt_pose_discriminator": trainer.opt_pose_discriminator.state_dict(),
        "opt_motion_discriminator": trainer.opt_motion_discriminator.state_dict(),
        "epoch": opt.train_batch
    }
    torch.save(state, opt.model_path + str(opt.train_batch) + ".tar")