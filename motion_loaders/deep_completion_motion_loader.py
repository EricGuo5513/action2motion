import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import utils.paramUtil as paramUtil
import models.deep_completion as models


class DeepCompletionGeneratedDataset(Dataset):

    def __init__(self, seq_length, input_size, num_labels,
                 dim_noise_pose, dim_noise_motion,
                 hidden_size_pose, hidden_size_motion,
                 model_path_pose, model_path_motion,
                 num_motions, batch_size, device):

        self.num_motions = num_motions

        print(num_labels)

        pose_models = torch.load(model_path_pose)
        pose_G = models.PoseGenerator(dim_noise_pose, num_labels, hidden_size_pose, input_size, device).to(device)
        pose_G.load_state_dict(pose_models['model_G'])
        pose_G.eval()

        motion_models = torch.load(model_path_motion)
        motion_G = models.MotionGenerator(dim_noise_pose, num_labels, dim_noise_motion, hidden_size_motion, seq_length, device).to(device)
        motion_G.load_state_dict(motion_models['model_G'])
        motion_G.eval()

        with torch.no_grad():
            motions_output = torch.zeros(num_motions, seq_length, input_size)
            labels_output = torch.zeros(num_motions)

            Tensor = torch.cuda.FloatTensor if device != 'cpu' else torch.FloatTensor
            while num_motions:
                num_motions_batch = min(batch_size, num_motions)

                categories_indices = torch.randint(num_labels, size=(num_motions_batch,))
                labels_output[(num_motions-num_motions_batch):num_motions] = categories_indices
                categories_one_hot = F.one_hot(categories_indices, num_classes=num_labels).float().to(device)

                z_0_batch = Tensor(num_motions_batch, dim_noise_pose).uniform_(0, 1).to(device)
                # dim(num_motions_batch, motion_len - 1, dim_noise_pose)
                fake_latent = motion_G(categories_one_hot, z_0_batch).to(device)
                # dim(num_motions_batch, 1, dim_noise_pose)
                z_0_batch = z_0_batch.unsqueeze_(1)
                # dim(num_motions_batch, motion_len, dim_noise_pose)
                motion_latent = torch.cat((z_0_batch, fake_latent), dim=1)
                #             for i in range(1, motion_latent.size(1)):
                #                 motion_latent[:, i, :] += motion_latent[:, i-1, :]

                for frame in range(seq_length):
                    motions_output[(num_motions-num_motions_batch):num_motions, frame, :] = \
                        pose_G(categories_one_hot, motion_latent[:, frame, :].squeeze())

                num_motions -= num_motions_batch

            self.motions_output = motions_output.numpy()
            self.labels_output = labels_output.numpy()

    def __len__(self):
        return self.num_motions

    def __getitem__(self, item):
        return self.motions_output[item, :, :], self.labels_output[item]


def get_deep_completion_motion_loader(num_motions, device):
    print('Generating Deep Completion Motion...')
    motion_gan_motion_dataset = DeepCompletionGeneratedDataset(60, 54, len(paramUtil.ntu_action_labels),
                                                               30, 30, 128, 256,
                                                               './model_file/deep_completion_pose.tar',
                                                               './model_file/deep_completion_motion.tar',
                                                               num_motions, 32, device)
    motion_loader = DataLoader(motion_gan_motion_dataset, batch_size=1, num_workers=1)
    return motion_loader
