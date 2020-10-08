from torch.utils.data import DataLoader
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from motion_loaders.motion_vae_motion_loader import MotionVAEGeneratedDataset
from motion_loaders.motion_gan_motion_loader import MotionGanGeneratedDataset
from motion_loaders.rnn_motion_loader import ConditionedRNNGeneratedDataset
from motion_loaders.deep_completion_motion_loader import DeepCompletionGeneratedDataset
from motion_loaders.motion_vae_lie_motion_loader import MotionVAELieGeneratedDataset
from motion_loaders.motion_vae_lie_veloc_motion_loader import MotionVAEVelocGeneratedDataset


def get_motion_loader(opt_path, num_motions, batch_size, device, ground_truth_motion_loader=None, label=None):
    opt = get_opt(opt_path, num_motions, device)

    if '/vae/' in opt_path:
        if 'veloc' in opt.name:
            print('Generating %s ...' % opt.name)
            dataset = MotionVAEVelocGeneratedDataset(opt, num_motions, batch_size, device, ground_truth_motion_loader, label)
        elif 'lie' in opt.name:
            print('Generating %s ...' % opt.name)
            dataset = MotionVAELieGeneratedDataset(opt, num_motions, batch_size, device, ground_truth_motion_loader, label)
        else:
            print('Generating Adversaried Motion VAE Motion...')
            # print(label)
            dataset = MotionVAEGeneratedDataset(opt, num_motions, batch_size, device, label)
    elif '/motion_gan' in opt_path:
        print('Generating Motion GAN Motion...')
        dataset = MotionGanGeneratedDataset(opt, opt.motion_length, opt.input_size_raw, len(opt.label_dec),
                                            opt.dim_z, opt.hidden_size, opt.model_file_path,
                                            num_motions, device)
    elif 'RNN' in opt_path:
        print('Generating Baseline - RNN Motion...')
        # print(opt.label_dec)
        dataset = ConditionedRNNGeneratedDataset(opt.motion_length, opt.input_size_raw, len(opt.label_dec),
                                                 ground_truth_motion_loader, opt.model_file_path_override,
                                                 num_motions, device)
    elif 'deep_completion' in opt_path:
        print('Generating Deep Completion Motion...')
        dataset = DeepCompletionGeneratedDataset(opt.motion_length, opt.input_size_raw, len(opt.label_dec),
                                                 opt.dim_noise_pose, opt.dim_noise_motion,
                                                 opt.hidden_size_pose, opt.hidden_size_motion,
                                                 opt.pose_file_path, opt.motion_file_path,
                                                 num_motions, 32, device)
    else:
        raise NotImplementedError('Unrecognized model type')

    motion_loader = DataLoader(dataset, batch_size=128, num_workers=1)
    return motion_loader
