from options.base_vae_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=100, help='Batch size of training process')

        self.parser.add_argument('--arbitrary_len', action='store_true', help='Enable variable length (batch_size has to'
                                                                              ' be 1 and motion_len will be disabled)')
        self.parser.add_argument('--do_adversary', action='store_true', help='Enable adversary motion discriminator')
        self.parser.add_argument('--do_recognition', action='store_true', help='Enable action recognition motion classifier')
        self.parser.add_argument('--do_align', action='store_true', help='Calculate align loss')
        self.parser.add_argument('--do_trajec_align', action='store_true', help='Calculate trajectory align loss')
        self.parser.add_argument('--optim_seperate', action='store_true', help='Calculate trajectory align loss')

        self.parser.add_argument('--use_geo_loss', action='store_true', help='Calculate align loss')
        self.parser.add_argument('--lambda_trajec', type=float, default=0.8, help='Calculate align loss')

        self.parser.add_argument('--skip_prob', type=float, default=0, help='Probability of skip frame while collecting loss')
        self.parser.add_argument('--tf_ratio', type=float, default=0.6, help='Teacher force learning ratio')

        self.parser.add_argument('--lambda_kld', type=float, default=0.0001, help='Weight of KL Divergence')
        self.parser.add_argument('--lambda_align', type=float, default=0.5, help='Weight of align loss')
        self.parser.add_argument('--lambda_adversary', type=float, default=0.5, help='Weight of adversary loss')
        self.parser.add_argument('--lambda_recognition', type=float, default=0.5, help='Weight of recognition loss')

        self.parser.add_argument('--do_kld_schedule', action='store_true', help='Activate KLD scheduler')
        self.parser.add_argument('--kld_schedule_range', type=str, default='0-0', help='From which iteration we '
                                                                                      'start increase kld weight')
        self.parser.add_argument('--end_lambda_kld', type=float, default=0.01, help='Termination of kld weight while scheduling')

        self.parser.add_argument('--is_continue', action="store_true", help='Enable continue training')
        self.parser.add_argument('--iters', type=int, default=20, help='Training iterations')

        self.parser.add_argument('--plot_every', type=int, default=500, help='Frequency of while plot loss curve')
        self.parser.add_argument("--save_every", type=int, default=500,
                            help='Frequency of saving intermediate models during training')
        self.parser.add_argument("--eval_every", type=int, default=500,
                                 help='Frequency of save intermediate samples during training')
        self.parser.add_argument("--save_latest", type=int, default=500,
                                 help='Frequency of saving latest models during training')
        self.parser.add_argument('--print_every', type=int, default=50, help='Frequency of printing training progress')
        self.isTrain = True
