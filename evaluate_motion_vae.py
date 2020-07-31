import models.motion_vae as vae_models
from trainer.vae_trainer import *
from utils.matrix_transformer import MatrixTransformer as mt
from utils.plot_script import plot_3d_motion, draw_pose_from_cords
import utils.paramUtil as paramUtil
from utils.utils_ import *
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
    result_path = os.path.join(opt.result_path, opt.dataset_type, opt.name + opt.name_ext)

    if opt.dataset_type == "humanact12":
        dataset_path = "./dataset/humanact12"
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
                                       opt.prior_hidden_layers, opt.num_samples, device)
    if opt.use_lie:
        decoder = vae_models.DecoderGRULie(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size,
                                           opt.decoder_hidden_layers,
                                           opt.num_samples, device)
    else:
        decoder = vae_models.DecoderGRU(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size,
                                        opt.decoder_hidden_layers,
                                        opt.num_samples, device)
    prior_net.load_state_dict(model['prior_net'])
    decoder.load_state_dict(model['decoder'])
    prior_net.to(device)
    decoder.to(device)
    if opt.use_lie:
        if opt.dataset_type=='shihao':
            data = dataset.MotionFolderDatasetShihaoV2(opt.clip_set, dataset_path, pkl_path, opt,
                                                       lie_enforce=opt.lie_enforce, raw_offsets=raw_offsets,
                                                       kinematic_chain=kinematic_chain)
        elif opt.dataset_type == 'humanact12':
            data = dataset.MotionFolderDatasetHumanAct12(dataset_path, opt, lie_enforce=opt.lie_enforce)
        elif opt.dataset_type == 'ntu_rgbd_vibe':
            data = dataset.MotionFolderDatasetNtuVIBE(file_prefix, motion_desc_file, labels, opt, joints_num=joints_num,
                                                      offset=True, extract_joints=paramUtil.kinect_vibe_extract_joints)
        elif opt.dataset_type == 'mocap':
            data = dataset.MotionFolderDatasetMocap(clip_path, dataset_path, opt)
        motion_dataset = dataset.MotionDataset(data, opt)
        motion_loader = DataLoader(motion_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=2,
                                   shuffle=True)
        trainer = TrainerLie(motion_loader, opt, device, raw_offsets, kinematic_chain)
    else:
        trainer = Trainer(None, opt, device)

    if opt.do_random:
        fake_motion, classes = trainer.evaluate(prior_net, decoder, opt.num_samples)
        fake_motion = fake_motion.cpu().numpy()
    else:
        categories = np.arange(opt.dim_category).repeat(opt.replic_times, axis=0)
        # categories = np.arange(1).repeat(opt.replic_times, axis=0)
        # categories = np.array([6]).repeat(opt.replic_times, axis=0)
        num_samples = categories.shape[0]
        category_oh, classes = trainer.get_cate_one_hot(categories)
        fake_motion, _ = trainer.evaluate(prior_net, decoder, num_samples, category_oh)
        fake_motion = fake_motion.cpu().numpy()

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
        np.save(os.path.join(keypoint_path, class_type + str(i) + '_3d.npy'), motion_mat)

        if opt.dataset_type == "shihao" or opt.dataset_type == "humanact12":
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
            '''
            motion_mat = mt.swap_yz(motion_mat)
            motion_mat[:, :, 2] *= -1
            plot_3d_motion(motion_mat, pose_tree, class_type, file_name, interval=150)
            '''
            ground_trajec = motion_mat[:, 0, :]
            plot_3d_motion_with_trajec(motion_mat, kinematic_chain, save_path=file_name, interval=80, trajec2=ground_trajec)

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
            # plot_3d_motion(motion_mat, pose_tree, class_type, file_name, interval=150)
            ground_trajec = motion_mat[:, 0, :]
            plot_3d_motion_with_trajec(motion_mat, kinematic_chain, save_path=file_name, interval=80,
                                       trajec1=ground_trajec, dataset="mocap")