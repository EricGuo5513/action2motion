import joblib
import utils.paramUtil as paramUtil
from utils.matrix_transformer import MatrixTransformer
from utils.plot_script import plot_3d_motion
import os

file_prefix = "../dataset/ntu_vibe/"
save_prefix = "../result/motion_gan/ntu_rgbd/vibe/ground_truth/"

for cla in paramUtil.ntu_action_labels:
    # filename = "S032C003P106R001A0" + str(cla) + ".pkl" if cla < 100 else "S032C003P106R001A" + str(cla) + ".pkl"
    # filename = "S032C003P106R001A00" + str(cla) + ".pkl" if cla < 10 else filename
    filename = "S017C003P020R001A0" + str(cla) + ".pkl" if cla < 100 else "S017C003P020R001A" + str(cla) + ".pkl"
    filename = "S017C003P020R001A00" + str(cla) + ".pkl" if cla < 10 else filename
    print(filename)
    action_id = int(filename[filename.index('A') + 1:-4])
    enumerator = paramUtil.ntu_action_enumerator
    class_type = enumerator[action_id]
    pose = None
    try:
        data = joblib.load(os.path.join(file_prefix, filename))
        print(os.path.join(file_prefix, filename))
        pose = data[1]['joints3d']
        print(pose.shape)
    except Exception as e:
        print(e)
        continue
    pose = pose[:, paramUtil.kinect_vibe_extract_joints, :]
    offset = pose[0][0]

    motion_mat = pose - offset
    motion_mat = motion_mat.reshape(-1, 18, 3)
    motion_mat = MatrixTransformer.swap_yz(motion_mat)
    motion_mat[:, :, 2] = -1 * motion_mat[:, :, 2]
    motion_mat = MatrixTransformer.rotate_along_z(motion_mat, 180)

    pose_tree = paramUtil.kinect_tree_vibe
    save_path = save_prefix + class_type + ".gif"
    plot_3d_motion(motion_mat, pose_tree, class_type, save_path, interval=150)