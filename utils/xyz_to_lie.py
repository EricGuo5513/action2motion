import os
from pathlib import Path
import torch
import numpy as np
import utils.general_utils as gu


index = []
index.append([0, 1, 2, 3])  # right leg
index.append([0, 6, 7, 8])  # left leg
index.append([0, 12, 13, 15])  # head
index.append([13, 17, 18, 19])  # left arm
index.append([13, 25, 26, 27])  # right arms


def xyz_to_lie_parameters(joint_xyz):
    num_joint = joint_xyz.shape[0]
    joint_xyz = joint_xyz - joint_xyz[0]
    lie_parameters = np.zeros([num_joint - 1, 6])
    # Location of joint 1 in chain
    for j in range(num_joint - 1):
        lie_parameters[j, 3] = np.linalg.norm(
            (joint_xyz[j, :] - joint_xyz[j + 1, :]))
    # Axis angle parameters of rotation
    for j in range(num_joint - 2, -1, -1):
        v = np.squeeze(joint_xyz[j + 1, :] - joint_xyz[[j], :])
        vhat = v / np.linalg.norm(v)
        if j == 0:
            uhat = [1, 0, 0]
        else:
            u = np.squeeze(joint_xyz[j, :] - joint_xyz[j - 1, :])
            uhat = u / np.linalg.norm(u)
        a = np.transpose(gu.rotmat(gu.findrot([1, 0, 0], uhat)))
        b = gu.rotmat(gu.findrot([1, 0, 0], vhat))
        c = gu.axis_angle(np.dot(a, b))
        lie_parameters[j, 0: 3] = c
    return lie_parameters


def convert(joint_xyz):
    num_frame = joint_xyz.shape[0]
    lie_parameters = np.zeros([joint_xyz.shape[0], 16, 6])
    for i in range(num_frame):
        joint_xyz[i, :, :] = joint_xyz[i, :, :] - joint_xyz[i, 0, :]
        for k in range(len(index)):
            lie_parameters[i, 3 * k + 1: 3 * k + 4, :] = xyz_to_lie_parameters(joint_xyz[i][index[k]])
    return lie_parameters


def main(config):
    if config.dataset == 'h36m':
        DATASET_NAME = config.dataset.lower()
        dataset_path = Path('data', DATASET_NAME, f'data_3d_{DATASET_NAME}.npz')
        dataset = np.load(dataset_path, allow_pickle=True)['positions_3d'].item()
        # joint_xyz = data['positions_3d'].item()['S1']['Walking 1']
        lie_data = {'positions_lie': {}}
        for subject in dataset.keys():
            if subject not in lie_data['positions_lie'].keys():
                lie_data['positions_lie'][subject] = {}
            for action in dataset[subject]:
                joint_xyz = dataset[subject][action]
                lie_parameters = convert(joint_xyz)
                lie_data['positions_lie'][subject][action] = lie_parameters
        np.savez('lie.npz', lie_data)


if __name__ == '__main__':
    config = parse_args()
    # os setting
    os.environ['OMP_NUM_THREAD'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

    # a simple work-around for the dataloader multithread bug
    torch.set_num_threads(1)
    main(config)
