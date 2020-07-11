import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="test", help='Name of experiment(For creating save root)')
        self.parser.add_argument("--gpu_id", type=str, default='0',
                                 help='Specify used gpu')

        self.parser.add_argument('--time_counter', action='store_true', help='Enable time count in generation')
        self.parser.add_argument('--coarse_grained', action='store_true', help='Enable time count in generation')

        self.parser.add_argument('--no_trajectory', action="store_true", help='Training iterations')
        self.parser.add_argument('--lie_enforce', action="store_true", help='Training iterations')
        self.parser.add_argument('--use_lie', action="store_true", help='Training iterations')

        self.parser.add_argument("--motion_length", type=int, default=60, help="Length of motion")
        self.parser.add_argument('--dataset_type', type=str, default='', help='Type of motion data')
        self.parser.add_argument('--clip_set', type=str, default="./dataset/pose_clip_full.csv", help='File path of clip data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/vae', help='Root path of intermediate file/models')

        self.parser.add_argument("--dim_z", type=int, default=30, help='Dimension of motion noise')
        self.parser.add_argument('--hidden_size', type=int, default=128, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--prior_hidden_layers', type=int, default=1, help='Layers of GRU in prior net')
        self.parser.add_argument('--posterior_hidden_layers', type=int, default=1, help='Layers of GRU in posterior net')
        self.parser.add_argument('--decoder_hidden_layers', type=int, default=2, help='Layers of GRU in decoder net')
        self.parser.add_argument('--veloc_hidden_layers', type=int, default=1, help='Layers of GRU in decoder net')
        self.parser.add_argument('--d_hidden_layers', type=int, default=2, help='Layers of GRU in discriminators')

        self.parser.add_argument('--use_vel_S', action="store_true", help="Use simple velocity estimation network")
        self.parser.add_argument('--use_vel_H', action="store_true", help="Use hierarchical velocity estimation network")
        self.parser.add_argument('--use_hdl', action="store_true",
                                 help="Use hierarchical dense layer")
        self.parser.add_argument('--do_all_parent', action="store_true",
                                 help="Use all parents in hdl")

        self.parser.add_argument('--do_relative', action="store_true",
                                 help="Use hierarchical velocity estimation network")

        self.parser.add_argument('--epoch_size', type=int, help="epoch_size of training velocity estimation network")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.isTrain = self.isTrain

        if self.opt.gpu_id != '':
            self.opt.gpu_id = int(self.opt.gpu_id)
            torch.cuda.set_device(self.opt.gpu_id)
        else:
            self.opt.gpu_id = None
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if self.isTrain:
            self.opt.kld_schedule_start = int(self.opt.kld_schedule_range.split('-')[0])
            self.opt.kld_schedule_end = int(self.opt.kld_schedule_range.split('-')[1])
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_type, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt




