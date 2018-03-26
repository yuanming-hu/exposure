import sys
import os
import matplotlib
from util import load_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from net import GAN


def main():
    config_name = sys.argv[1]
    cfg = load_config(config_name)
    cfg.name = sys.argv[1] + '/' + sys.argv[2]
    net = GAN(cfg, restore=False)
    net.train()


if __name__ == '__main__':
    main()
