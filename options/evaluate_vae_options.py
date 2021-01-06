from options.base_vae_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default="latest", help='The epoch need evluation')
        self.parser.add_argument('--result_path', type=str, default="./eval_results/vae/", help='Save path of evaluate results')
        self.parser.add_argument('--replic_times', type=int, default=1, help='Replication times of all categories')
        self.parser.add_argument('--do_random', action='store_true', help='Random generation')
        self.parser.add_argument('--num_samples', type=int, default=100, help='Number of generated')
        self.parser.add_argument('--batch_size', type=int, default=20, help='Batch size of training process')

        self.parser.add_argument('--name_ext', type=str, default="", help='Extension of save path')

        # for ablation study only
        self.parser.add_argument('--save_latent', action='store_true', help='Save latent vector')
        self.parser.add_argument('--start_step', type=int, default=30, help='Start step where the latent stop being fixed')

        # for interpolation only
        self.parser.add_argument('--do_interp', action='store_true', help='Do interpolation study')
        self.parser.add_argument('--do_quantile', action='store_true', help='Do quantitle of latent vector')

        self.parser.add_argument('--interp_step', type=int, default=0,
                                 help='Step where interpolation happens')
        self.parser.add_argument('--interp_bins', type=int, default=10,
                                 help='Step to which the latent vector is interpolated')
        self.parser.add_argument('--interp_type', type=str, default="linear",
                                 help='Step to which the latent vector is interpolated')

            # for quantile only
        self.parser.add_argument('--pp_dims', type=str, default='-1',
                                 help='Step to which the latent vector is interpolated')

        # for action shift only
        self.parser.add_argument('--do_action_shift', action='store_true', help='Do action shift generation')
        self.parser.add_argument('--action_list', type=str, default="0,1",
                                 help='Step to which the latent vector is interpolated')
        self.parser.add_argument('--shift_steps', type=str, default="50",
                                 help='Step to which the latent vector is interpolated')
        self.isTrain = False