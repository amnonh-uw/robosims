from util import train_reg
from util.config import parse_args

def train(argv):
    args = parse_args(argv)
    train_regression(args, conf.args.model)

if __name__ == "__main__":
    train(sys.argv[1:])
