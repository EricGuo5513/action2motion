import numpy as np
from utils.plot_script import *
import utils.paramUtil as paramUtil
from utils.utils_ import *

# For motion inpainting in CMU Mocap
def main1():
    data_prefix = "./eval_results/diverse/mocap/vae_velocS_f0001_t01_trj10_rela_run10_s20/keypoint/Run"
    id_list = ['8', '6', '3', '5', '10', '19']
    motion_list = [np.load(data_prefix + id_s + '_3d.npy') for id_s in id_list]
    kinematic_chain = paramUtil.mocap_kinematic_chain
    plot_3d_multi_motion(motion_list, kinematic_chain,
                         data_prefix + "_mul.gif",
                         interval=80, dataset='mocap')

# For motion inpainting in HumanAct13
def main2():
    data_prefix = "./eval_results/diverse/humanact13/vae_velocS_f0001_t001_trj10_rela_jump47_s10/keypoint/jump"
    id_list = ['0', '10', '5']
    motion_list = [np.load(data_prefix + id_s + '_3d.npy') for id_s in id_list]
    kinematic_chain = paramUtil.shihao_kinematic_chain
    plot_3d_multi_motion(motion_list, kinematic_chain,
                         data_prefix + "_mul.gif",
                         interval=100, dataset='humanact13')

# For variation at each step
def main3():
    data_prefix = "./eval_results/shift/humanact13/vae_velocS_f0001_t001_trj10_rela_sit_jump_dumb_s30_60_l100/keypoint/sit_jump_lift_dumbbell11"
    joints_path = data_prefix + "_3d.npy"
    logvar_path = data_prefix + "_lgvar.npy"

    img_path = "./eval_results/shift/humanact13/vae_velocS_f0001_t001_trj10_rela_sit_jump_dumb_s30_60_l100/sit_jump_lift_dumbbell11"
    kinematic_chain = paramUtil.shihao_kinematic_chain
    joints_3d = np.load(joints_path)
    logvar = np.load(logvar_path)
    frames = joints_3d.shape[0]
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in range(frames):
        pose_3d = joints_3d[i]
        pose_lgvar = logvar[i].mean()
        plot_3d_pose_v2(os.path.join(img_path, str(i)+".png"), kinematic_chain, pose_3d, title= "LogVar%.3f" % pose_lgvar)

# For latent interpolation
def main4():
    data_prefix = "/home/chuan/language2pose/action2pose/eval_results/quantile/humanact13/vae_velocS_f0001_t001_trj10_rela_jump3_s20_b20/"
    name = "jump"
    path_list = ["%s%s%d.png" % (data_prefix, name, i) for i in range(20)]
    img_list = []

    img_size = (250, 300)
    for path in path_list:
        img = Image.open(path)
        img_arr = np.asarray(img)
        img_crop = img_arr[100:400, 200:450]
        img_list.append(img_crop)
    compose_and_save_img(img_list, data_prefix, "lift_dumbbell_transition.png", 10, 2, img_size)


# For batch post-processing(smoothing) of motions
def main5():
    data_path = "./eval_results/vae/humanact13/vae_velocS_f0001_t001_trj10_rela_fineG/"
    keypoint_path = os.path.join(data_path, "keypoint")
    file_list = os.listdir(keypoint_path)
    file_list = [item for item in file_list if "3d" in item]
    file_list.sort()
    kinematic_chain = paramUtil.shihao_kinematic_chain
    save_dir = os.path.join(data_path, "smooth_3d")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, file_name in enumerate(file_list):
        print("%d / %d" % (i, len(file_list)))
        joints_3d_path = os.path.join(keypoint_path, file_name)
        joints_3d = np.load(joints_3d_path)

        motion_name = file_name[:file_name.rfind('_')]

        smooth_3d = temporal_filter(joints_3d, sigma=1.5).reshape(joints_3d.shape)

        np.save(os.path.join(save_dir, motion_name + "_3ds.npy"), smooth_3d)
        ground_trajec = smooth_3d[:, 0, :]

        # print(motion_name)
        plot_3d_motion_with_trajec(smooth_3d, kinematic_chain, save_path=os.path.join(data_path, motion_name + ".gif")
                                   , interval=80, trajec1=ground_trajec)
        # break


if __name__ == "__main__":
    main5()