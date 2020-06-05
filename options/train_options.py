from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--motion_batch', type=int, default=3, help='Batch size of motion discriminator')
        self.parser.add_argument('--pose_batch', type=int, default=30, help='Batch size of pose discriminator')
        self.parser.add_argument('--iterations', type=int, default=20, help='Training iterations')
        self.parser.add_argument('--use_categories', action="store_true", help='Batch size of motion discriminator')
        self.parser.add_argument('--use_smoothness', action="store_true", help='Batch size of pose discriminator')
        self.parser.add_argument('--use_pose_discriminator', action="store_true", help='Training iterations')
        self.parser.add_argument('--use_wgan', action="store_true", help='Training iterations')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')

        self.parser.add_argument("--save_every", type=int, default=500,
                            help='Frequency of saving generated samples during training')
        self.parser.add_argument("--eval_every", type=int, default=500,
                                 help='Frequency of saving generated samples during training')
        self.parser.add_argument("--save_latest", type=int, default=500,
                                 help='Frequency of saving generated samples during training')
        self.parser.add_argument('--print_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--plot_every', type=int, default=500, help='Frequency of plot points')
        self.isTrain = True
