import argparse
import yaml
from easydict import EasyDict

def parse_args(argv):
    parser = argparse.ArgumentParser(description='a3c train')
    parser.add_argument('--load-model', dest='load_model', action='store_true')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.add_argument('--server-config', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--test-only', dest='test_only', action='store_true')
    parser.set_defaults(load_model=False)
    parser.set_defaults(test_only=False)

    args = parser.parse_args(argv)
    args.conf = config()
    if args.config != None:
        args.conf.load(args.config)

    return args

class config(EasyDict):
    def __init__(self):
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
        self.discrete_action_rotation = 1

        # criteria for goal success
        self.close_enough_distance = 0.01
        self.close_enough_rotation = 1

        # distance between source and destination
        self.max_distance_delta = 1
        self.max_rotation_delta = 3
        self.probe_distance = 1

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
        self.base_class = 'GoogleNet'
        self.load_base_weights = True

        # loss
        self.entropy_loss_weight = 10.

        self.learning_rate = 1e-4

    def load(self, file_name):
        with open(file_name, 'r') as stream:
            y = yaml.safe_load(stream)
            for k, v in y.items():
                assert k in self, '{} is not a valid config parameter.'.format(k)
                assert type(v) == type(self[k]), 'type of parameter {} is not matched'.format(k)
    
            self[k] = v
