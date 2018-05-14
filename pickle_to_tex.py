# This script converts output pickle to step-by-step latex figures

import numpy as np
import os
import pickle as pickle
import shutil

NUM_STEPS = 5
CURVE_STEPS = 8

files = []
filters = [
    'Expo.',
    'Gam.',
    'W.B.',
    'Satu.',
    'Tone',
    'Cst.',
    'BW',
    'Color',
]


def visualize_detail(name, param, pos):

  def map_pos(x, y):
    return '(%f,%f)' % (pos[0] + x * 0.8, pos[1] - 1.1 + y * 0.8)

  if name == 'Expo.':
    return '{Exposure $%+.2f$};' % param[0]
  elif name == 'Gam.':
    return '{Gamma $1/%.2f$};' % (1 / param[0])
  elif name == 'Satu.':
    return '{Saturation $+%.2f$};' % param[0]
  elif name == 'Cst.':
    return '{Contrast $%+.2f$};' % param[0]
  elif name == 'BW':
    return '{$%+.2f$};' % (param[0])
  elif name == 'W.B.':
    scaling = 1 / (1e-5 + 0.27 * param[0] + 0.67 * param[1] + 0.06 * param[2])
    r, g, b = [int(255 * x * scaling) for x in param]
    color = r'{\definecolor{tempcolor}{RGB}{%d,%d,%d}};' % (r, g, b)
    return color + '\n' + r'\tikz \fill[tempcolor] (0,0) rectangle (4 ex, 2 ex);'
  elif name == 'Tone':
    s = '{Tone\quad\quad\quad\quad};\n'
    s += r'\draw[<->] %s -- %s -- %s;' % (map_pos(0, 1.1), map_pos(0, 0),
                                          map_pos(1.1, 0))
    s += '\n'

    for i in range(1):
      values = np.array([0] + list(param[0][0][i]))
      values /= sum(values) + 1e-30
      scale = 1
      values *= scale
      for j in range(0, CURVE_STEPS):
        values[j + 1] += values[j]

      for j in range(CURVE_STEPS):
        p1 = (1.0 / CURVE_STEPS * j, values[j])
        p2 = (1.0 / CURVE_STEPS * (j + 1), values[j + 1])
        s += r'\draw[-] %s -- %s;' % (map_pos(*p1), map_pos(*p2))
        if j != CURVE_STEPS - 1:
          s += '\n'
    return s
  elif name == 'Color':
    s = '{Color\quad\quad\quad\quad};\n'
    s += r'\draw[<->] %s -- %s -- %s;' % (map_pos(0, 1.1), map_pos(0, 0),
                                          map_pos(1.1, 0))
    s += '\n'

    c = ['red', 'green', 'blue']
    for i in range(3):
      #print(param)
      values = np.array([0] + list(param[0][0][i]))
      values /= sum(values) + 1e-30
      scale = 1
      values *= scale
      for j in range(0, CURVE_STEPS):
        values[j + 1] += values[j]

      for j in range(CURVE_STEPS):
        p1 = (1.0 / CURVE_STEPS * j, values[j])
        p2 = (1.0 / CURVE_STEPS * (j + 1), values[j + 1])
        s += r'\draw[%s,-] %s -- %s;' % (c[i], map_pos(*p1), map_pos(*p2))
        if j != CURVE_STEPS - 1:
          s += '\n'
    return s
  else:
    assert False


def visualize_step(debug_info, step_name, position):
  pdf = debug_info['pdf']
  filter_id = debug_info['selected_filter_id']
  s = ''
  s += r'\node[draw, rectangle, thick,minimum height=7em,minimum width=7em](%s) at (%f,%f) {};' % (
      step_name, position[0], position[1])
  s += '\n'
  s += r'\node (%ss) at ([yshift=1.4em]%s.center) {' % (step_name, step_name)
  s += '\n'
  s += r'    \scalebox{0.7}{'
  s += '\n'
  s += r'    \begin{tabular}{|p{0.5cm}p{0.2cm}p{0.5cm}p{0.2cm}|}'
  s += '\n'
  s += r'        \hline'
  s += '\n'

  def bar(i):
    return '\pdfbarSelected' if i == filter_id else '\pdfbar'

  for i in range(4):
    f1 = filters[i]
    b1 = r'%s{%.3f}' % (bar(i), pdf[i] * 3)
    f2 = filters[i + 4]
    b2 = r'%s{%.3f}' % (bar(i + 4), pdf[i + 4] * 3)
    s += r'        %s & %s & %s & %s \\' % (f1, b1, f2, b2)
    s += '\n'
  s += r'        \hline'
  s += '\n'
  s += r'    \end{tabular}'
  s += '\n'
  s += r'    }'
  s += '\n'
  s += r'};'
  s += '\n'
  s += r'\node (%sd) at ([yshift=-2.0em]%s.center)' % (step_name, step_name)
  s += '\n'
  s += visualize_detail(
      filters[filter_id],
      debug_info['filter_debug_info'][filter_id]['filter_parameters'], position)
  s += '\n'
  return s


def process_dog():
  f = 'dog04/a0694.tif_debug.pkl'
  debug_info_list = pickle.load(open(f, 'r'))

  for i in range(NUM_STEPS):
    debug_info = debug_info_list[i]
    print(visualize_step(debug_info, 'agent%d' % (i + 1), (4, i * -3)), end=' ')


def process(filename, id, src):
  pkl_fn = os.path.join(src, filename)
  debug_info_list = pickle.load(open(pkl_fn, 'rb'))
  filename = filename[:-10]
  target_dir = 'export/{}'.format(id)
  os.makedirs(target_dir, exist_ok=True)
  for i in range(NUM_STEPS - 1):
    shutil.copy(os.path.join(src, filename + '.intermediate%02d.png' % i),
                os.path.join(target_dir, 'step%d.png' % (i + 1)))

  shutil.copy(os.path.join(src, filename + '.retouched.png'), os.path.join(target_dir, 'final.png'))
  shutil.copy(os.path.join(src, filename + '.linear.png'), os.path.join(target_dir, 'input.png'))

  with open(target_dir + '/steps.tex', 'w') as f:
    for i in range(NUM_STEPS):
      debug_info = debug_info_list[i]
      print(
          visualize_step(debug_info, 'agent%d' % (i + 1), (4, i * -3)),
          end=' ',
          file=f)


print('##########################################')
print('Note: Please make sure you have pdflatex.')
print('##########################################')
print()

for input_dir in ['outputs']:

  for f in os.listdir(input_dir):
    if not f.endswith('pkl'):
      continue
    id = f.split('.')[0]
    print('Generating pdf operating sequences for image {}...'.format(id))
    process(f, id, src=input_dir)

