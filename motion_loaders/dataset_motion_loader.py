import utils.paramUtil as paramUtil
from dataProcessing import dataset
from torch.utils.data import DataLoader, RandomSampler


class Options:
    def __init__(self, lie_enforce, use_lie, no_trajectory, motion_length, coarse_grained):
        self.lie_enforce = lie_enforce
        self.use_lie = use_lie
        self.no_trajectory = no_trajectory
        self.motion_length = motion_length
        self.coarse_grained = coarse_grained
        self.save_root = './model_file/'
        self.clip_set = './dataset/pose_clip_full.csv'

cached_dataset = {}


def get_dataset_motion_dataset(opt, label=None):
    if opt.opt_path in cached_dataset:
        return cached_dataset[opt.opt_path]

    if opt.dataset_type == "humanact12":
        dataset_path = "./dataset/humanact12"
        HumanAct12Options = Options(False, False, False, 60, True)
        data = dataset.MotionFolderDatasetHumanAct12(dataset_path, opt, lie_enforce=opt.lie_enforce)
        motion_dataset = dataset.MotionDataset(data, HumanAct12Options)
    elif opt.dataset_type == 'ntu_rgbd_vibe':
        file_prefix = "./dataset"
        motion_desc_file = "ntu_vibe_list.txt"
        joints_num = 18
        labels = paramUtil.ntu_action_labels
        NtuVIBEOptions = Options(False, True, False, 60, True)

        data = dataset.MotionFolderDatasetNtuVIBE(file_prefix, motion_desc_file, labels, NtuVIBEOptions,
                                                  joints_num=joints_num, offset=True,
                                                  extract_joints=paramUtil.kinect_vibe_extract_joints)
        motion_dataset = dataset.MotionDataset(data, NtuVIBEOptions)
    elif opt.dataset_type == "shihao":
        dataset_path = "./dataset/pose"
        pkl_path = './dataset/pose_shihao_merge'
        ShihaoOptions = Options(False, False, False, 60, True)
        raw_offsets = paramUtil.shihao_raw_offsets
        kinematic_chain = paramUtil.shihao_kinematic_chain
        data = dataset.MotionFolderDatasetShihaoV2(opt.clip_set, dataset_path, pkl_path, opt,
                                                   lie_enforce=opt.lie_enforce, raw_offsets=raw_offsets,
                                                   kinematic_chain=kinematic_chain)
        motion_dataset = dataset.MotionDataset(data, ShihaoOptions)
    elif opt.dataset_type == "mocap":
        dataset_path = "./dataset/mocap/mocap_3djoints/"
        clip_path = './dataset/mocap/pose_clip.csv'
        mocap_options = Options(False, False, False, 100, True)
        data = dataset.MotionFolderDatasetMocap(clip_path, dataset_path, opt)
        # print(label)
        if label is None:
            motion_dataset = dataset.MotionDataset(data, mocap_options)
        else:
            motion_dataset = dataset.MotionDataset4One(data, mocap_options, label)
            # print(len(motion_dataset))
    else:
        raise NotImplementedError('Unrecognized dataset')

    cached_dataset[opt.opt_path] = motion_dataset
    return motion_dataset



def get_dataset_motion_loader(opt, num_motions, device, label=None):
    print('Generating Ground Truth Motion...')
    motion_dataset = get_dataset_motion_dataset(opt, label)
    # print(len(motion_dataset))
    motion_loader = DataLoader(motion_dataset, batch_size=1, num_workers=1,
                               sampler=RandomSampler(motion_dataset, replacement=True, num_samples=num_motions))

    return motion_loader
