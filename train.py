from util.train_regression import train_regression
from util.config import parse_args

def train(argv):
    args = parse_args(argv)
    train_regression(args, args.conf.model)

if __name__ == "__main__":
    train(sys.argv[1:])
