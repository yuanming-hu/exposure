import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from net import GAN
from util import load_config


def evaluate():
  if len(sys.argv) < 4:
    print(
        "Usage: python3 evaluate.py [task name] [model name] [image files name1] [image files name2] ..."
    )
    exit(-1)
  if len(sys.argv) == 4:
    print(
        " Note: Process a single image at a time may be inefficient - try multiple inputs)"
    )
  print("(TODO: batch processing when images have the same resolution)")
  print()
  print("Initializing...")
  config_name = sys.argv[1]
  import shutil
  shutil.copy('models/%s/%s/scripts/config_%s.py' %
              (config_name, sys.argv[2], config_name), 'config_tmp.py')
  cfg = load_config('tmp')
  cfg.name = sys.argv[1] + '/' + sys.argv[2]
  net = GAN(cfg, restore=True)
  net.restore(20000)
  spec_files = sys.argv[3:]
  print('processing files {}', spec_files)
  net.eval(spec_files=spec_files)


if __name__ == '__main__':
  evaluate()
