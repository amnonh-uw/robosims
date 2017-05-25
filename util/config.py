import argparse
import yaml
import re
import os
from easydict import EasyDict

def parse_args(argv):
    parser = argparse.ArgumentParser(description='a3c train')
    parser.add_argument('--load-model', dest='load_model', action='store_true')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.add_argument('--partial-load-model', type=str, default=None)
    parser.add_argument('--server-config', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--test-only', dest='test_only', action='store_true')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--postfix', type=str, default=None)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--base-class', type=str, default=None)
    parser.set_defaults(load_model=False)
    parser.set_defaults(test_only=False)

    args = parser.parse_args(argv)
    args.conf = config()
    if args.config != None:
        args.conf.load(args.config)
        if args.conf.postfix is None:
            args.conf.postfix = os.path.splitext(os.path.basename(args.config))[0]

    args.conf.server_config = args.server_config
    args.conf.gen_dataset = False

    if args.base_class != None:
        args.conf.base_class = args.base_class

    if args.iter == 0 and args.conf.iter != 0:
        args.iter = args.conf.iter
    else:
        args.conf.iter = args.iter

    if args.epochs == 0 and args.conf.epochs != 0:
        args.epochs = args.conf.epochs
    else:
        args.conf.epochs = args.epochs

    if args.dataset == None and args.conf.dataset != None:
        args.dataset = args.conf.dataset
    else:
        args.conf.dataset = args.dataset

    if args.test_only:
        if args.conf.epochs != 0 and args.conf.epochs != 1:
            print('--test-only must have 1 epoch')
            exit()
        args.conf.epochs = 1

        args.conf.test_only = True
        if args.conf.iter == 0:
            args.conf.test_iter = 100
        else:
            conf.test_iter = args.iter
    else:
        if args.conf.iter == 0:
            args.conf.iter = 40000
        if args.conf.epochs == 0:
            args.conf.epochs = 10

    # if postfix is not set, create one if there is a config file
    if args.conf.postfix == None and args.config != None:
        args.conf.postfix = re.sub('^.*/', '', re.sub('\.yaml$', '', args.config))
        print("defaulting to postfix {}".format(args.conf.postfix))

    return args

class config(EasyDict):
    def __init__(self):
        self.postfix = None
        self.dataset = None
        self.testset = None
        self.test_steps = 0
        self.cheat = False
        self.check_gradients = True
        self.iter = 0
        self.test_iter = 100
        self.epochs = 0
        self.model_save_frequency = 100 # batches
        self.flush_plot_frequency = 20 # batches
        self.averages_period = 20 # batches
        self.batch_size = 32
        self.log_device_placement = False
        self.allow_soft_placement = True
        self.colocate_gradients_with_ops = True
        self.use_adam = True
        self.relative_errors = False
        self.verify_dataset = None
        self.verify_iter = 0
        self.verify_frequencey = 500000
        self.pose_dims = 3
        self.highlight_absolute_error = 0.1
        self.highlight_relative_error = 0.25


        # a3c
        self.num_workers = 1
        self.max_episode_length = 100
        self.episode_buffer_size = 10

        # image resolution
        self.h_size = 400
        self.v_size = 300
        self.channels = 3

        self.sensors = 1
        self.gamma = 0.99

        # dicrete grid
        self.discrete_action_distance = 0.1
        self.discrete_action_rotation = 1.

        # criteria for goal success
        self.close_enough_distance = 0.01
        self.close_enough_rotation = 1.

        # distance between source and destination
        self.max_distance_delta = 0.1
        self.max_rotation_delta = 3.
        self.probe_distance = 1.

        # rewards
        self.collision_reward = -1000
        self.step_reward = -1
        self.close_enough_reward = 1000

        # action types
        self.discrete_actions = True
        self.continous_actions = False

        # grid
        self.grid_distance = 0.01
        self.grid_rotation = 0.1
        self.bfs_grid_x = 0.5
        self.bfs_grid_y = 0.5
        self.bfs_grid_z = 0.5

        # heuristic
        self.reward_heuristic_weight = 0.

        # gif
        self.gif_fps = 20
        self.base_class = 'vgg16'
        self.load_base_weights = False

        # loss
        self.error_clip_min = None
        self.error_clip_max = None
        # self.loss_clip_min = 0.1        # 10%
        # self.loss_clip_max = 1.1        # 110%
        self.entropy_loss_weight = 500.
        self.value_loss_weight = 0.5
        self.policy_loss_weight = 0.5

        self.learning_rate = 1e-4

    def load(self, file_name):
        with open(file_name, 'r') as stream:
            y = yaml.safe_load(stream)
            for k, v in y.items():
                if isinstance(self[k], float) and isinstance(v, int):
                    v = float(v)
                assert k in self, '{} is not a valid config parameter.'.format(k)
                assert (type(v) == type(self[k])) or (self[k] == None), 'type of parameter {} is not matched: {} {}'.format(k, type(v), type(self[k]))
    
                self[k] = v
