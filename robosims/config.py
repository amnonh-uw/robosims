import os
import os.path as osp
from easydict import EasyDict as edict

CONFIG = edict()

CONFIG.ENV_TYPE     = "living-room"
CONFIG.ENV_NAME     = "living-room-0"
CONFIG.ENV_BUILD_DARWIN    = "unity/builds/living-room-OSXIntel64.app"
CONFIG.ENV_BUILD_LINUX    = "unity/builds/living-room-Linux64"
CONFIG.PROCESS_NAME = "Robot AI Platform"
CONFIG.PREFIX       = "ROBOSIMS_"
CONFIG.X_DISPLAY       = "0.0"

CONFIG.TASK_TYPE   = "navigation"
CONFIG.TASK_TARGET = "balcony"

# during training, make the agent go faster
CONFIG.TRAIN_WALK_VELOCITY = 20.0
CONFIG.TRAIN_TURN_VELOCITY = 100.0
CONFIG.TRAIN_ACTION_LENGTH = 4

# during test, make the agent go slower
CONFIG.TEST_WALK_VELOCITY = 2.0
CONFIG.TEST_TURN_VELOCITY = 10.0
CONFIG.TEST_ACTION_LENGTH = 40

CONFIG.AGENT_HEIGHT = 0.05
CONFIG.AGENT_RADIUS = 0.1

CONFIG.COMPUTE_DEPTH_MAP  = False
CONFIG.HUMAN_CONTROL_MODE = False
CONFIG.TRAIN_PHASE        = False
CONFIG.SERVER_SIDE_SCREENSHOT = False

def stringify_config(c):
    # turn values into strings
    return dict((CONFIG.PREFIX + x, str(c[x])) for x in c)

def get_config():
    # return current configurations
    return stringify_config(CONFIG)

def update_config_from_yaml(filename):
    # Overwrite default options by parameters specified in the yaml file
    import yaml
    with open(filename, 'r') as f:
        config = edict(yaml.load(f))

    for k, v in config.items():
        assert k in CONFIG, '{} is not a valid config parameter.'.format(k)
        assert type(v) == type(CONFIG[k]), 'type of parameter {} is not matched'.format(k)
        # update config parameters
        CONFIG[k] = v

    return stringify_config(CONFIG)

def pretty_print_config():
    import json
    print(json.dumps(CONFIG, indent=2))
