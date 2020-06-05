from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default="latest", help='Frequency of plot points')
        self.parser.add_argument('--result_path', type=str, default="./eval_results/", help='Frequency of plot points')
        self.parser.add_argument('--replic_times', type=int, default=1, help='Frequency of plot points')
        self.parser.add_argument('--pose_batch', type=int, default=30, help='Batch size of pose discriminator')
        self.parser.add_argument('--use_lie', action="store_true", help='Frequency of plot points')

        self.isTrain = False