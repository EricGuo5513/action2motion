from torch.utils.data import DataLoader

import utils.paramUtil as paramUtil
from trainer.vae_trainer import *
from models.networks import *
from dataProcessing import dataset
from utils.plot_script import plot_loss, print_current_loss
from options.evaluate_vae_options import TestOptions
import os


motion_names = [
    "P05G01R04F0642T0778A0201.npy",
    "P05G03R04F0426T0457A1201.npy",
    "P06G01R01F0580T0649A0301.npy",
    "P06G01R01F0711T0767A0401.npy",
    "P06G01R01F0780T0815A0402.npy",
    "P07G01R01F0631T0767A0301.npy",
    "P07G01R01F0430T0569A0201.npy"
]

label_enc_rev = {
    '01': 0,
    '07': 1,
    '06': 2,
    '09': 3,
    '08': 4,
    '05': 5,
    '11': 6,
    '12': 7,
    '10': 8,
    '04': 9,
    '03': 10,
    '02': 11
}

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


def motion_enumerator():
    motion_list = [np.load("./dataset/humanact13/" + motion).reshape(-1, 72) for motion in motion_names]
    labels = [label_enc_rev[motion[motion.find('A')+1: motion.find('.')-2]] for motion in motion_names]

    motion_list = [np.expand_dims(motion_arr, 0) for motion_arr in motion_list]
    labels = [np.ones((1, 1), dtype=np.int) * label for label in labels]

    return motion_list, labels


if __name__ == "__main__":
    parser = TestOptions()
    opt = parser.parse()
    device = torch.device("cuda:" + str(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    opt.checkpoint_root = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name)
    opt.model_path = os.path.join(opt.checkpoint_root, 'model')

    opt.save_path = os.path.join(opt.result_path, opt.dataset_type, opt.name)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    dataset_path = ""
    joints_num = 0
    input_size = 72
    # data = None

    # if opt.dataset_type == "humanact13":
    #     dataset_path = "./dataset/humanact13"
    #     input_size = 72
    #     joints_num = 24
    #     data = dataset.MotionFolderDatasetHumanAct13(dataset_path, opt, False, True)
    # elif opt.dataset_type == "mocap":
    #     dataset_path = "./dataset/mocap/mocap_3djoints/"
    #     clip_path = './dataset/mocap/pose_clip.csv'
    #     input_size = 60
    #     joints_num = 20
    #     raw_offsets = paramUtil.mocap_raw_offsets
    #     kinematic_chain = paramUtil.mocap_kinematic_chain
    #     data = dataset.MotionFolderDatasetMocap(clip_path, dataset_path, opt)
    # elif opt.dataset_type == "ntu_rgbd_vibe":
    #     file_prefix = "./dataset"
    #     motion_desc_file = "ntu_vibe_list.txt"
    #     joints_num = 18
    #     input_size = 54
    #     labels = paramUtil.ntu_action_labels
    #     raw_offsets = paramUtil.vibe_raw_offsets
    #     kinematic_chain = paramUtil.vibe_kinematic_chain
    #     data = dataset.MotionFolderDatasetNtuVIBE(file_prefix, motion_desc_file, labels, opt, joints_num=joints_num,
    #                                           offset=True, extract_joints=paramUtil.kinect_vibe_extract_joints)
    # else:
    #     raise NotImplementedError('This dataset is unregonized!!!')

    opt.dim_category = 12
    opt.input_size = input_size * 2 + opt.dim_category
    opt.output_size = 3

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    motion_batches, label_batches = motion_enumerator()

    if opt.use_vel_S:
        model = VelocityNetwork_Sim(opt.input_size, opt.output_size, opt.hidden_size)
    else:
        model = VelocityNetwork(opt.input_size, opt.output_size, opt.hidden_size, 1, opt.batch_size, device)

    load_network(model, opt.model_path, 'latest.tar')

    model.to(device)
    model.eval()

    for i in range(len(motion_batches)):
        print("%2d / %2d" % (i, len(motion_batches)))
        offset = motion_batches[i][0, 0, :3]
        motion = motion_batches[i][0] - np.tile(offset, 24)
        motion = motion.reshape(-1, 24, 3)
        ground_trajectory = motion_batches[i][0, :, :3] - offset
        # print(ground_trajectory)

        data = torch.from_numpy(motion_batches[i]).to(device)
        labels = label_batches[i]
        cate_ones, _ = get_cate_one_hot(labels)
        data = Tensor(data.size()).copy_(data)

        # print(data.shape)
        data = data - data[..., :3].repeat(1, 1, 24)
        data1 = data[:, 1:, :]
        data2 = data[:, :-1, :]
        cate_ones = cate_ones.unsqueeze(1).repeat(1, data1.shape[1], 1)
        inputs = torch.cat((cate_ones, data1, data2), dim=-1)
        # inputs (batch_size, motion_len - 1, 72)
        inputs = Tensor(inputs.size()).copy_(inputs)

        pred_trajectory = np.zeros((1, 3))
        for k in range(inputs.shape[1]):
            pred_velocity = model(inputs[:, k, :]).detach().cpu().numpy()
            pred_velocity = pred_velocity / 10
            pred_trajectory = np.concatenate((pred_trajectory, pred_trajectory[-1] + pred_velocity), axis=0)

        name = motion_names[i].split('.')[0]
        # plot_3d_trajectory(pred_trajectory, os.path.join(opt.save_path, name + '.png'), ground_trajectory)
        plot_3d_motion_with_trajec(motion, paramUtil.shihao_kinematic_chain, os.path.join(opt.save_path, name + '.gif'),
                                   interval=80, trajec1=pred_trajectory, trajec2=ground_trajectory)