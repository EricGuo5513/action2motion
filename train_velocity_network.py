import argparse
import torch
from torch.utils.data import DataLoader

import models.motion_gan as gan_models
import models.motion_vae as vae_models
import utils.paramUtil as paramUtil
from trainer.vae_trainer import *
from models.networks import *
from dataProcessing import dataset
from utils.plot_script import plot_loss, print_current_loss
from options.train_vae_options import TrainOptions
import os


def get_cate_one_hot(categories):
    classes_to_generate = np.array(categories).reshape((-1,))
    # dim (num_samples, dim_category)
    one_hot = np.zeros((categories.shape[0], opt.dim_category), dtype=np.float32)
    one_hot[np.arange(categories.shape[0]), classes_to_generate] = 1

    # dim (num_samples, dim_category)
    one_hot_motion = torch.from_numpy(one_hot).to(device).requires_grad_(False)

    return one_hot_motion, classes_to_generate


def save_network(network, save_path, save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, save_name)
    torch.save(network.state_dict(), save_path)


def load_network(network, save_path, save_name):
    save_path = os.path.join(save_path, save_name)
    params = torch.load(save_path)
    network.load_state_dict(params)


if __name__ == "__main__":
    parser = TrainOptions()
    opt = parser.parse()
    device = torch.device("cuda:" + str(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    opt.save_root = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name)
    opt.model_path = os.path.join(opt.save_root, 'model')

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    dataset_path = ""
    joints_num = 0
    input_size = 72
    data = None

    if opt.dataset_type == "humanact13":
        dataset_path = "./dataset/humanact13"
        input_size = 72
        joints_num = 24
        data = dataset.MotionFolderDatasetHumanAct13(dataset_path, opt, False, True)
    elif opt.dataset_type == "mocap":
        dataset_path = "./dataset/mocap/mocap_3djoints/"
        clip_path = './dataset/mocap/pose_clip.csv'
        input_size = 60
        joints_num = 20
        raw_offsets = paramUtil.mocap_raw_offsets
        kinematic_chain = paramUtil.mocap_kinematic_chain
        data = dataset.MotionFolderDatasetMocap(clip_path, dataset_path, opt)
    elif opt.dataset_type == "ntu_rgbd_vibe":
        file_prefix = "./dataset"
        motion_desc_file = "ntu_vibe_list.txt"
        joints_num = 18
        input_size = 54
        labels = paramUtil.ntu_action_labels
        raw_offsets = paramUtil.vibe_raw_offsets
        kinematic_chain = paramUtil.vibe_kinematic_chain
        data = dataset.MotionFolderDatasetNtuVIBE(file_prefix, motion_desc_file, labels, opt, joints_num=joints_num,
                                              offset=True, extract_joints=paramUtil.kinect_vibe_extract_joints)
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    opt.dim_category = len(data.labels)
    opt.input_size = input_size * 2 + opt.dim_category
    opt.output_size = 3

    if opt.use_vel_S:
        motion_dataset = dataset.PairFrameDataset(data)
        motion_loader = DataLoader(motion_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=2,
                                   shuffle=True)
        model = VelocityNetwork_Sim(opt.input_size, opt.output_size, opt.hidden_size)
    else:
        motion_dataset = dataset.MotionDataset(data, opt)
        motion_loader = DataLoader(motion_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=2,
                                   shuffle=True)
        model = VelocityNetwork(opt.input_size, opt.output_size, opt.hidden_size, 1, opt.batch_size, device)

    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), weight_decay=0.00001)

    if opt.is_continue:
        load_network(model, opt.model_path, 'latest.tar')

    model.to(device)
    model.train()

    def __init_log__():
        log = OrderedDict()
        log['total_loss'] = []
        return log

    total_steps = 1
    loss_log = __init_log__()
    start_time = time.time()
    niter_per_epo = len(motion_loader)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    pc_model = sum(param.numel() for param in model.parameters())
    print(model)
    print("Total parameters of prior net: {}".format(pc_model))

    print("# training dataset size {0}".format(len(motion_dataset)))
    print("# number of iterations per epoch {0}".format(niter_per_epo))
    for epoch in range(opt.epoch_size):

        for niter, batch in enumerate(motion_loader):

            optimizer.zero_grad()
            if opt.use_vel_S:
                # data1 (batch_size, num_joints * 3)
                (data1, data2, labels) = batch
                data1 = Tensor(data1.size()).copy_(data1)
                data2 = Tensor(data2.size()).copy_(data2)
                # amplify the error by 10 times
                ground = (data2[..., :3] - data1[..., :3]) * 10
                cate_ones, _ = get_cate_one_hot(labels)
                data1 = data1 - data1[..., :3].repeat(1, joints_num)
                data2 = data2 - data2[..., :3].repeat(1, joints_num)

                inputs = torch.cat((cate_ones, data1, data2), dim=1)
                inputs = Tensor(inputs.size()).copy_(inputs).detach()
                ground = Tensor(ground.size()).copy_(ground).detach()

                output = model(inputs)

                total_loss = mse(output, ground)
                loss_log['total_loss'].append(total_loss.item())
            else:
                data, labels = batch
                # data (batch_size, motion_len, num_joints*3)
                data = Tensor(data.size()).copy_(data)
                # ground (batch_size, motion_len - 1, 3)
                ground = (data[:, 1:, :3] - data[:, :-1, :3]) * 10
                data = data - data[..., :3].repeat(1, 1, joints_num)
                data1 = data[:, 1:, :]
                data2 = data[:, :-1, :]
                # cate_ones (batch_size, cate_dim)
                cate_ones, _ = get_cate_one_hot(labels)
                # cate_ones (batch_size, motion_len-1, cate_dim)
                cate_ones = cate_ones.unsqueeze(1).repeat(1, data1.shape[1], 1)
                inputs = torch.cat((cate_ones, data1, data2), dim=-1)
                inputs = Tensor(inputs.size()).copy_(inputs).detach()
                ground = Tensor(ground.size()).copy_(ground).detach()
                model.init_hidden()
                total_loss = 0

                for i in range(inputs.shape[1]):
                    h_in = inputs[:, i, :]
                    h_out = model(h_in)
                    total_loss += mse(h_out, ground[:, i, :])
                loss_log['total_loss'].append(total_loss.item() / inputs.shape[1])

            total_loss.backward()
            optimizer.step()

            if total_steps % opt.print_every == 0:
                mean_loss = OrderedDict()
                for k, v in loss_log.items():
                    mean_loss[k] = sum(loss_log[k][-1 * opt.print_every:]) / opt.print_every
                print_current_loss(start_time, total_steps, opt.epoch_size * niter_per_epo, mean_loss, epoch, niter)
            if total_steps % opt.save_latest == 0:
                save_network(model, opt.model_path, 'latest.tar')
            total_steps += 1

    save_network(model, opt.model_path, 'latest.tar')
    plot_loss(loss_log, os.path.join(opt.save_root, "loss_curve.png"), intervals=opt.plot_every)