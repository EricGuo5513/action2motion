import numpy as np
import torch
import utils.paramUtil as paramUtil
from models.rnn_model import ConditionedRNN
from torch.utils.data import Dataset, DataLoader
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader


class ConditionedRNNGeneratedDataset(Dataset):
    def __init__(self, seq_length, input_size, num_labels, ground_truth_motion_loader,
                 model_path, num_motions, device):

        self.num_motions = num_motions

        model = torch.load(model_path)
        rnn_generator = ConditionedRNN(num_labels, input_size, 128, input_size).to(device)
        rnn_generator.load_state_dict(model['model'])
        rnn_generator.eval()

        self.motions_output = torch.zeros(num_motions, seq_length, input_size)
        self.labels_output = torch.zeros(num_motions)
        with torch.no_grad():
            motion_loader_iter = iter(ground_truth_motion_loader)

            for idx in range(num_motions):
                motion, label = next(motion_loader_iter)
                model_input = torch.clone(motion[0]).float().detach_().to(device)
                category_tensor = np.zeros((1, num_labels))
                category_tensor[0, label] = 1
                category_tensor = torch.tensor(category_tensor, dtype=torch.float, device=device)

                motion_output = torch.zeros(model_input.shape[0], model_input.shape[1])
                motion_output[0, :] = model_input[0, :]
                model_hidden = rnn_generator.initHidden()
                for i in range(motion.shape[0] - 1):
                    model_output, model_hidden = rnn_generator(
                        model_input, model_hidden, category_tensor
                    )
                    motion_output[i] = model_output
                    model_input = model_output

                self.motions_output[idx, :, :] = motion_output
                self.labels_output[idx] = label

    def __len__(self):
        return self.num_motions

    def __getitem__(self, item):
        return self.motions_output[item, :, :], self.labels_output[item]
