from __future__ import division, print_function, absolute_import

import os

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string('data_fp', None, 'data filepath')

FLAGS = flags.FLAGS
logging.set_verbosity(logging.DEBUG)

def load_data(fp):
  logging.fatal('Unimplemented')

def plot_heatmap(array, y_labels, x_labels):
  if len(array.shape) != 2:
    logging.fatal('Input array must have 2 dimensions')
  y_dim, x_dim = array.shape
  if y_dim != len(y_labels):
    logging.fatal('y dim of input array does not match labels')
  if x_dim != len(x_labels):
    logging.fatal('x dim of input array does not match labels')

  plt.imshow(array, cmap='Blues')
  plt.xticks(np.arange(x_dim), x_labels)
  plt.yticks(np.arange(y_dim), y_labels)
  plt.tick_params(labeltop=True, labelbottom=False, length=0)
  for j in range(y_dim):
    for i in range(x_dim):
      plt.text(i, j, '{0:.2f}'.format(array[j, i]), ha='center', va='center', fontsize=14)


def main(unused_argv):
  del unused_argv

  if FLAGS.data_fp:
    logging.info('Loading data from {}'.format(FLAGS.data_fp))
    data, y_labels, x_labels = load_data(FLAGS.data_fp)
  else:
    logging.info('Generating sample data')
    data = np.random.random(size=(2, 10))
    y_labels = ['y1', 'y2']
    x_labels = ['x' + str(i) for i in range(10)]

  plot_heatmap(data, y_labels, x_labels)
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  app.run(main)
