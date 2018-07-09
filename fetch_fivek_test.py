import sys
import os
from util import read_set
import shutil

five_k_dataset_path = 'data/fivek_dataset/'

try:
  os.makedirs(os.path.join(five_k_dataset_path, 'test_set'))
except:
  pass

def fetch():
  for i in read_set('u_test'):
    fn = "{:04d}.tif".format(i)
    shutil.copy(os.path.join(five_k_dataset_path, 'FiveK_Lightroom_Export_InputDayLight', fn),
                os.path.join(five_k_dataset_path, 'test_set', fn))
    
if __name__ == '__main__':
  fetch()
