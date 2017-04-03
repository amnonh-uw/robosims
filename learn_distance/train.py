from learn_distance.distance_network import *
from networks.run_train import *

def train(argv):
    args = parse_args(argv)
    run_train(args, Distance_Model)

if __name__ == "__main__":
    train(sys.argv[1:])
