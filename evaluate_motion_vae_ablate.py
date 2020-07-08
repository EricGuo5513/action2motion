import models.motion_vae as vae_models
import models.networks as networks
from trainer.vae_trainer import *
from utils.matrix_transformer import MatrixTransformer as mt
from utils.plot_script import plot_3d_motion, draw_pose_from_cords
import utils.paramUtil as paramUtil
from options.evaluate_vae_options import *
from dataProcessing import dataset
from torch.utils.data import DataLoader

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

    latent_path = ""
    motion_path = ""

    latents = np.load(latent_path)
    motion = np.load(motion_path)
    latents = torch.from_numpy(latents).float().to(device)
    motion = torch.from_numpy(motion).float().to(device)
    motion_mat = motion.unsqueeze_(0)

    action_indx = 0

    if opt.dataset_type == "humanact13":
        dataset_path = "./dataset/humanact13"
        input_size = 72
        joints_num = 24
        label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        raw_offsets = paramUtil.shihao_raw_offsets
        kinematic_chain = paramUtil.shihao_kinematic_chain
        enumerator = paramUtil.shihao_coarse_action_enumerator
    elif opt.dataset_type == "shihao":
        dataset_path = "./dataset/pose"
        pkl_path = './dataset/pose_shihao_merge'
        input_size = 72
        joints_num = 24
        label_dec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        raw_offsets = paramUtil.shihao_raw_offsets
        kinematic_chain = paramUtil.shihao_kinematic_chain
        enumerator = paramUtil.shihao_coarse_action_enumerator
    elif opt.dataset_type == "ntu_rgbd":
        file_prefix = "./dataset/"
        motion_desc_file = "motionlist.txt"
        joints_num = 25
        input_size = 75
        labels = paramUtil.ntu_action_labels
    elif opt.dataset_type == "ntu_rgbd_v2":
        file_prefix = "./dataset/"
        motion_desc_file = "motionlistv2.txt"
        joints_num = 19
        input_size = 57
        labels = paramUtil.ntu_action_labels
    elif opt.dataset_type == "mocap":
        dataset_path = "./dataset/mocap/mocap_3djoints/"
        clip_path = './dataset/mocap/pose_clip.csv'
        input_size = 60
        joints_num = 20
        raw_offsets = paramUtil.mocap_raw_offsets
        kinematic_chain = paramUtil.mocap_kinematic_chain
        label_dec = [0, 1, 2, 3, 4, 5, 6, 7]
        enumerator = paramUtil.mocap_action_enumerator
    elif opt.dataset_type == "ntu_rgbd_vibe":
        file_prefix = "./dataset"
        motion_desc_file = "ntu_vibe_list.txt"
        joints_num = 18
        input_size = 54
        label_dec = [6, 7, 8, 9, 22, 23, 24, 38, 80, 93, 99, 100, 102]
        labels = paramUtil.ntu_action_labels
        enumerator = paramUtil.ntu_action_enumerator
        raw_offsets = paramUtil.vibe_raw_offsets
        kinematic_chain = paramUtil.vibe_kinematic_chain
    else:
        raise NotImplementedError('This dataset is unregonized!!!')

    opt.dim_category = len(label_dec)

    opt.pose_dim = input_size

    if opt.time_counter:
        opt.input_size = input_size + opt.dim_category + 1
    else:
        opt.input_size = input_size + opt.dim_category

    opt.output_size = input_size

    model = torch.load(model_file_path)
    prior_net = vae_models.GaussianGRU(opt.input_size, opt.dim_z, opt.hidden_size,
                                       opt.prior_hidden_layers, opt.batch_size, device)

    if opt.use_vel_S:
        veloc_net = networks.VelocityNetwork_Sim(input_size * 2 + 20, 3, opt.hidden_size)
    elif opt.use_vel_H:
        veloc_net = networks.VelocityNetworkHierarchy(3, kinematic_chain)
    else:
        veloc_net = networks.VelocityNetwork(input_size * 2 + 20, 3, opt.hidden_size, opt.veloc_hidden_layers,
                                             opt.batch_size, device)

    decoder = vae_models.DecoderGRULieV2(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size,
                                         opt.decoder_hidden_layers, opt.batch_size, device, use_hdl=opt.use_hdl,
                                         do_all_parent=opt.do_all_parent, kinematic_chains=kinematic_chain)

    prior_net.load_state_dict(model['prior_net'])
    veloc_net.load_state_dict(model['veloc_net'])
    decoder.load_state_dict(model['decoder'])

    prior_net.to(device)
    decoder.to(device)
    veloc_net.to(device)

    if opt.do_relative:
        trainer = TrainerLieV3(None, opt, device, raw_offsets, kinematic_chain)
    else:
        trainer = TrainerLieV2(None, opt, device, raw_offsets, kinematic_chain)

    dim_category = len(data.labels)

    categories = np.array([action_indx,]).repeat(opt.num_samples, axis=0)
    category_oh, classes = trainer.get_cate_one_hot(categories)

    fake_motion, _, latent_batch, logvar = trainer.evaluate_4_manip(prior_net, decoder, veloc_net, opt.num_samples, latents,
                                                       opt.start_step, cate_one_hot=category_oh, real_joints=motion_mat)
    fake_motion = fake_motion.numpy()
    latent_batch = latent_batch.numpy()
    logvar = logvar.numpy()

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
        '''
        offset = np.matlib.repmat(np.array([motion_orig[0, 0], motion_orig[0, 1], motion_orig[0, 2]]),
                                     motion_orig.shape[0], joints_num)
        # offset = np.tile(motion_orig[:, :3], 24)
        # print(offset[1])
        motion_mat = motion_orig - offset

        '''
        motion_mat = motion_orig

        motion_mat = motion_mat.reshape(-1, joints_num, 3)
        # motion_mat[:, :, 2] *= -1
        np.save(os.path.join(keypoint_path, class_type + str(i) + '_3d.npy'), motion_mat)

        if opt.dataset_type == "shihao" or opt.dataset_type == "humanact13":
            pose_tree = paramUtil.smpl_tree
            ground_trajec = motion_mat[:, 0, :]
            plot_3d_motion_with_trajec(motion_mat, kinematic_chain, save_path=file_name, interval=80, trajec1=ground_trajec)

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
            '''
            motion_mat = mt.swap_yz(motion_mat)
            motion_mat[:, :, 2] = -1 * motion_mat[:, :, 2]
            motion_mat = mt.rotate_along_z(motion_mat, 180)
            pose_tree = paramUtil.kinect_tree_vibe
            # exclued_points = paramUtil.excluded_joint_ids
            plot_3d_motion(motion_mat, pose_tree, class_type, file_name, interval=150)
            '''
            plot_3d_motion_v2(motion_mat, kinematic_chain, save_path=file_name, interval=80)
        elif opt.dataset_type == "mocap":
            pose_tree = paramUtil.kinect_tree_mocap
            ground_trajec = motion_mat[:, 0, :]
            plot_3d_motion_with_trajec(motion_mat, kinematic_chain, save_path=file_name, interval=80,
                                       trajec1=ground_trajec, dataset="mocap")