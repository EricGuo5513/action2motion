import numpy as np
from utils.plot_script import *
import utils.paramUtil as paramUtil
from utils.utils_ import *


def main1():
    data_prefix = "./eval_results/diverse/mocap/vae_velocS_f0001_t01_trj10_rela_run10_s20/keypoint/Run"
    id_list = ['8', '6', '3', '5', '10', '19']
    motion_list = [np.load(data_prefix + id_s + '_3d.npy') for id_s in id_list]
    kinematic_chain = paramUtil.mocap_kinematic_chain
    plot_3d_multi_motion(motion_list, kinematic_chain,
                         data_prefix + "_mul.gif",
                         interval=80, dataset='mocap')


def main2():
    data_prefix = "./eval_results/diverse/humanact13/vae_velocS_f0001_t001_trj10_rela_jump47_s10/keypoint/jump"
    id_list = ['0', '10', '5']
    motion_list = [np.load(data_prefix + id_s + '_3d.npy') for id_s in id_list]
    kinematic_chain = paramUtil.shihao_kinematic_chain
    plot_3d_multi_motion(motion_list, kinematic_chain,
                         data_prefix + "_mul.gif",
                         interval=100, dataset='humanact13')


def main3():
    data_prefix = ""
    kinematic_chain = paramUtil.shihao_kinematic_chain

if __name__ == "__main__":
    main1()