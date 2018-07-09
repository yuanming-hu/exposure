import random
import os

with open('data/artists/filmA_test.txt') as f:
  inputs = list(map(int, f.readlines()))

random.shuffle(inputs)

files = sorted(os.listdir('data/artists/filmAHDR'))
for i in inputs:
  print('data/artists/filmAHDR/{}'.format(files[i]), end=' ')

