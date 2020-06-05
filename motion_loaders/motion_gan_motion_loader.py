import torch
import models.motion_gan as models
from trainer.gan_trainers import Trainer
from torch.utils.data import Dataset


class MotionGanGeneratedDataset(Dataset):
    def __init__(self, opt, seq_length, input_size, num_labels, dim_noise, hidden_size,
                 model_path, num_motions, device):

        self.num_motions = num_motions

        model = torch.load(model_path)
        generator = models.MotionGenerator(dim_noise, num_labels, seq_length,
                                           hidden_size, opt, input_size=input_size, output_size=input_size).to(device)
        generator.load_state_dict(model['generator'])

        fixed_m_noise = generator.sample_z_r(num_motions)
        labels_one_hot, labels_output = generator.sample_z_categ(num_motions)

        motions_output = Trainer.evaluate(generator, fixed_m_noise, labels_one_hot, num_motions).numpy()

        self.motions_output = motions_output
        self.labels_output = labels_output

    def __len__(self):
        return self.num_motions

    def __getitem__(self, item):
        return self.motions_output[item, :, :], self.labels_output[item]
