import torch
import models.motion_vae as vae_models
from trainer.vae_trainer import Trainer
from torch.utils.data import Dataset
import numpy as np


class MotionVAEGeneratedDataset(Dataset):

    def __init__(self, opt, num_motions, batch_size, device, label=None):
        self.num_motions = num_motions
        self.label = label

        model = torch.load(opt.model_file_path)
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

        trainer = Trainer(None, opt, device)

        if opt.do_random:
            motions_output = torch.zeros(num_motions, opt.motion_length, opt.input_size_raw).numpy()
            labels_output = torch.zeros(num_motions).numpy()

            while num_motions:
                num_motions_batch = min(batch_size, num_motions)

                opt.num_samples = num_motions_batch
                cate_one_hot = None
                if self.label is not None:
                    categories = np.ones(opt.num_samples, dtype=np.int)
                    categories.fill(self.label)
                    cate_one_hot, _ = trainer.get_cate_one_hot(categories)
                # print(self.label)
                motions_output_batch, labels_output_batch = \
                    trainer.evaluate(prior_net, decoder, opt.num_samples, cate_one_hot=cate_one_hot)
                if self.label is not None:
                    labels_output_batch = self.label
                motions_output[(num_motions-num_motions_batch):num_motions, :, :] = motions_output_batch
                labels_output[(num_motions-num_motions_batch):num_motions] = labels_output_batch

                num_motions -= num_motions_batch

        else:
            raise NotImplementedError('LOL, not today!')

        self.motions_output = motions_output
        self.labels_output = labels_output

    def __len__(self):
        return self.num_motions

    def __getitem__(self, item):
        return self.motions_output[item, :, :], self.labels_output[item]
