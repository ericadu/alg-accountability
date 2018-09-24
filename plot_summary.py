from __future__ import division, print_function, absolute_import

import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

flags.DEFINE_string('sys1', None, 'System 1')
flags.DEFINE_string('sys2', None, 'System 2')
flags.DEFINE_string('exp', None, 'Experiment name')

FLAGS = flags.FLAGS
logging.set_verbosity(logging.DEBUG)

def get_filepath(sys_name, exp_name):
  return 'data/counterfactual/{}/{}_results.csv'.format(sys_name, exp_name)

def load_data(fp1, fp2):
  df1 = pd.read_csv(fp1)
  df2 = pd.read_csv(fp2)

  parameter1 = df1.columns[0]
  parameter2 = df2.columns[0]

  df1 = df1.sort_values(by=[parameter1])
  df2 = df2.sort_values(by=[parameter2])

  if parameter1 != parameter2:
    logging.fatal('Mismatched parameters. Check files.')

  sys1_values = df1['FP'].values / 100
  sys2_values = df2['FP'].values / 100

  data = np.stack((sys1_values, sys2_values))
  x_labels = list(df1[parameter1].values)
  title = parameter1

  return data, x_labels, title

def plot_heatmap(array, y_labels, x_labels):
  if len(array.shape) != 2:
    logging.fatal('Input array must have 2 dimensions')
  y_dim, x_dim = array.shape
  if y_dim != len(y_labels):
    logging.fatal('y dim of input array does not match labels')
  if x_dim != len(x_labels):
    logging.fatal('x dim of input array does not match labels')

  plt.imshow(array, cmap='Blues', vmin=0, vmax=1.0)
  plt.xticks(np.arange(x_dim), x_labels)
  plt.yticks(np.arange(y_dim), y_labels)
  plt.tick_params(labeltop=True, labelbottom=False, length=0)
  for j in range(y_dim):
    for i in range(x_dim):
      plt.text(i, j, '{0:.2f}'.format(array[j, i]), ha='center', va='center', fontsize=14)

def main(unused_argv):
  del unused_argv

  if FLAGS.exp and FLAGS.sys1 and FLAGS.sys2:
    fp1 = get_filepath(FLAGS.sys1, FLAGS.exp)
    fp2 = get_filepath(FLAGS.sys2, FLAGS.exp)
    logging.info('Loading data from {} and {}'.format(fp1, fp2))
    data, x_labels, title = load_data(fp1, fp2)
    y_labels = [FLAGS.sys1, FLAGS.sys2]
  else:
    logging.info('Generating sample data')
    data = np.random.random(size=(2, 10))
    y_labels = ['y1', 'y2']
    x_labels = ['x' + str(i) for i in range(10)]
    title = 'example'

  plot_heatmap(data, y_labels, x_labels)
  plt.title(title, y=1.2)
  plt.tight_layout()
  plt.savefig(FLAGS.exp+ '.png')


if __name__ == '__main__':
  app.run(main)