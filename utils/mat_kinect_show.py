import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as sio
import utils.paramUtil as paramUtil
from utils.plot_script import plot_3d_pose, plot_3d_motion, plot_2d_motion
from utils.matrix_transformer import MatrixTransformer as mt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------Kinect Skeleton base------------------------------------------------------
# 	0 -  base of the spine
# 	1 -  middle of the spine
# 	2 -  neck
#	3 -  head
# 	4 -  left shoulder
# 	5 -  left elbow
# 	6 -  left wrist
# 	7 -  left hand
# 	8 -  right shoulder
# 	9 - right elbow
# 	10 - right wrist
# 	11 - right hand
# 	12 - left hip
# 	13 - left knee
# 	14 - left ankle
# 	15 - left foot
# 	16 - right hip
# 	17 - right knee
# 	18 - right ankle
# 	19 - right foot
# 	20 - spine
#	21 - tip of the left hand
# 	22 - left thumb
# 	23 - tip of the right hand
# 	24 - right thumb
# --------------------------------------------------------------------------------------------------------------

mpl.rcParams['legend.fontsize'] = 10
file_prefix = "../dataset/ntu_rgbd/nturgbd_s018_s032_mat/"
save_prefix = "../result/motion_gan/ntu_rgbd/v2/ground_truth/"

# for cla in range(1, 121):
    # filename = "S017C001P020R002A0" + str(cla) + ".mat" if cla < 100 else "S017C001P020R002A0"+ str(cla) +".mat"
    # filename = "S017C001P020R002A00" + str(cla) + ".mat" if cla < 10 else filename
# for cr in range(5, 12):
#     C_id = cr % 4
#     R_id = int(cr / 4)
for cla in paramUtil.ntu_action_labels:
    filename = "S018C001P043R002A0" + str(cla) + ".mat" if cla < 100 else "S018C001P043R002A"+ str(cla) +".mat"
    filename = "S018C001P043R002A00" + str(cla) + ".mat" if cla < 10 else filename
    print(filename)
    action_id = int(filename[filename.index('A') + 1:-4])
    enumerator = paramUtil.ntu_action_enumerator
    class_type = enumerator[action_id]
    try:
        data = np.array(sio.loadmat(file_prefix + filename)['joints']).astype(float)
        print(data.shape)
    except Exception:
        continue

    '''
    pose_orig = data[1]
    offset = numpy.matlib.repmat(np.array([pose_orig[0], pose_orig[1], pose_orig[2]]), 1, 25)[0]
    
    pose = pose_orig - offset
    
    body_entity = paramUtil.KinectBodyPart()
    
    plot_3d_pose(pose, body_entity)
    '''
    # save_path_3d = '../dataset/ntu_rgbd/' + class_type + "_C" + str(C_id) + "_R" + str(R_id) + '_3d_mt16.gif'
    # save_path_2d = '../dataset/ntu_rgbd/' + class_type + "_C" + str(C_id) + "_R" + str(R_id) + '_2d_mt16.gif'

    offset = numpy.matlib.repmat(np.array([data[0, 0], data[0, 1], data[0, 2]]), data.shape[0], 25)

    motion_mat = data - offset
    motion_mat = motion_mat.reshape(-1, 25, 3)
    motion_mat = mt.swap_yz(motion_mat)
    # motion_mat = mt.swap_xy(motion_mat)
    pose_tree = paramUtil.kinect_tree_exclude
    exclude_joints = paramUtil.excluded_joint_ids
    save_path = save_prefix + class_type + "2.gif"
    plot_3d_motion(motion_mat, pose_tree, class_type, save_path, interval=150, excluded_joints=exclude_joints)
    # motion_mat[:, :, 0] = motion_mat[:, :, 0] * -1
    # plot_2d_motion(motion_mat, pose_tree, 2, 0, class_type, save_path_2d, interval=150)
