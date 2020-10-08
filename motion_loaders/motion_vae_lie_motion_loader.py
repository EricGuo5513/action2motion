import torch
import models.motion_vae as vae_models
from trainer.vae_trainer import *
from torch.utils.data import Dataset
import utils.paramUtil as paramUtil


class MotionVAELieGeneratedDataset(Dataset):
    def __init__(self, opt, num_motions, batch_size, device, ground_motion_loader, label=None):
        if opt.dataset_type == 'ntu_rgbd_vibe':
            raw_offsets = paramUtil.vibe_raw_offsets
            kinematic_chain = paramUtil.vibe_kinematic_chain
        elif opt.dataset_type == 'shihao' or opt.dataset_type == 'humanact12':
            raw_offsets = paramUtil.shihao_raw_offsets
            kinematic_chain = paramUtil.shihao_kinematic_chain
        elif opt.dataset_type == 'mocap':
            raw_offsets = paramUtil.mocap_raw_offsets
            kinematic_chain = paramUtil.mocap_kinematic_chain
        else:
            raise NotImplementedError("Data type not Found")
        model = torch.load(opt.model_file_path)
        self.prior_net = vae_models.GaussianGRU(opt.input_size, opt.dim_z, opt.hidden_size,
                                           opt.prior_hidden_layers, opt.num_samples, device)

        self.decoder = vae_models.DecoderGRULie(opt.input_size + opt.dim_z, opt.output_size, opt.hidden_size,
                                               opt.decoder_hidden_layers,
                                               opt.num_samples, device)

        self.prior_net.load_state_dict(model['prior_net'])
        self.decoder.load_state_dict(model['decoder'])
        self.prior_net.to(device)
        self.decoder.to(device)
        self.num_motions = num_motions
        self.trainer = TrainerLie(None, opt, device, raw_offsets, kinematic_chain)

        self.pool_size = num_motions
        self.resize_counter = 0
        self.opt = opt
        self.batch_size = batch_size
        self.ground_motion_loader = ground_motion_loader
        self.label = label
        self.initiatize(self.opt, self.pool_size, self.batch_size, self.ground_motion_loader)

    def initiatize(self, opt, num_motions, batch_size, ground_motion_loader):
        self.resize_counter += 1
        #print_opt = vars(opt)
        #for k, v in sorted(print_opt.items()):
        #    print("%s: %s" % (k, v))
        if opt.do_random:
            motions_output = torch.zeros(num_motions, opt.motion_length, opt.input_size_raw).numpy()
            labels_output = torch.zeros(num_motions).numpy()
            real_iter = iter(ground_motion_loader)
            while num_motions:
                num_motions_batch = min(batch_size, num_motions)
                real_joints_list = []
                while len(real_joints_list) < num_motions_batch:
                    real_joints_list.append(next(real_iter)[0])
                real_joints = torch.cat(real_joints_list, dim=0)
                opt.num_samples = num_motions_batch

                cate_one_hot = None
                if self.label is not None:
                    categories = np.ones(opt.num_samples, dtype=np.int)
                    categories.fill(self.label)
                    cate_one_hot, _ = self.trainer.get_cate_one_hot(categories)
                motions_output_batch, labels_output_batch = \
                    self.trainer.evaluate(self.prior_net, self.decoder, opt.num_samples, cate_one_hot=cate_one_hot,
                                          real_joints=real_joints)
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
        if (item / self.pool_size) >= self.resize_counter:
            self.initiatize(self.opt, self.pool_size, self.batch_size, self.ground_motion_loader)
        item = item % self.pool_size
        return self.motions_output[item, :, :], self.labels_output[item]
