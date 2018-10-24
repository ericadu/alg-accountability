import os
from absl import app
from absl import flags
from absl import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
import statistical_parity_generator as sp

flags.DEFINE_string('file', '/Users/erica/Desktop/GitHub/data/compas_concat.csv', 'Input file.')
flags.DEFINE_integer('runs', None, 'Number of trials.')
flags.DEFINE_integer('threads', None, 'Number of threads.')

FLAGS = flags.FLAGS
logging.set_verbosity(logging.DEBUG)

def plot():
  df = pd.read_csv('output/propublica.csv')
  pl = df.boxplot(column='epsilon', by='delta')
  fig = pl.get_figure()
  fig.savefig("output/propublica.png")

def run(trial):
  with open('output/propublica.csv', 'a') as f:
    for delta in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]: #, 0.15, 0.2, 0.25, 0.3]:
      remove_n = int(delta * size / 2)
      a_df = df[df.race == 'African-American']
      c_df = df[df.race == 'Caucasian']

      # SUBSET
      # drop_indices = np.random.choice(c_df.index, remove_n, replace=False)
      # df_subset = c_df.drop(drop_indices)
      # df_subset = pd.concat([df_subset, a_df])

      # GLOBAL
      # drop_indices = np.random.choice(df.index, remove_n, replace=False)
      # df_subset = df.drop(drop_indices)

      # TARGETED
      condition = a_df.score_text == 'High'
      a_h_df = a_df[condition]
      drop_indices = np.random.choice(a_h_df.index, remove_n, replace=False)
      df_subset_a = a_df.drop(drop_indices)
      

      condition_c = c_df.score_text == 'Low'
      c_h_df = c_df[condition_c]
      drop_indices = np.random.choice(c_h_df.index, remove_n, replace=False)
      df_subset_c = c_df.drop(drop_indices)

      df_subset = pd.concat([df_subset_a, df_subset_c])

      gb = df_subset.groupby(['race', 'score_text']).size()
      r_a = gb['African-American'].sum() if 'African-American' in gb else 0
      r_c = gb['Caucasian'].sum() if 'Caucasian' in gb else 0

      high_a = 0
      high_c = 0

      if r_a > 0:
        high_a = gb['African-American']['High'] / r_a
      
      if r_c > 0:
        high_c = gb['Caucasian']['High'] / r_c

      eps = abs(high_a - high_c)
      results = [r_a, r_c, delta, eps]
      f.write(','.join([str(r) for r in results]) + '\n')
      #print("black: {}, white: {}, delta: {}, epsilon: {}".format(r_a, r_c, delta, abs(high_a - high_c)))


def main(unused_argv):
  del unused_argv
  global df
  global size

  num_trials = FLAGS.runs
  filename = FLAGS.file
  df = pd.read_csv(filename)
  size = df.index.size

  pool = Pool(FLAGS.threads)
  pool.map(run, list(range(num_trials)))
  pool.close() 
  pool.join()

if __name__ == '__main__':
  app.run(main)