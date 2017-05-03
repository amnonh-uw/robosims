from learn_direction.direction_network import *
from util.train_regression import *
from util.config import *

def train(argv):
    args = parse_args(argv)
    train_regression(args, Direction_Model)

if __name__ == "__main__":
    train(sys.argv[1:])
