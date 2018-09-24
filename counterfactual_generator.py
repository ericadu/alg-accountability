import numpy as np
import pandas as pd
import itertools as it

'''
Dataset generator satisfying epsilon counterfactual fairness. All dataset values are binary
and satisfy statistical parity.
========================
INPUTS
m       : int     [1, )   :   number of additional attributes
n       : int     [1, )   :   number of samples
biased  : boolean         :   unbiased or biased columns
eps     : float   [0, 1]  :   | Pr[ Y^ = 1 | A = 1, X ] - Pr[ Y^ = 1 | A = 0, X ] | < epsilon 
p       : float   [0, 1]  :   Pr[ X_i = 1 | Y^ = 1] (biased) or Pr[X_i = 1] (unbiased) 
========================
ALGORITHM
1. Populate outcome Y^ with probability 0.5
2. Populate protected attribute A with protected attribute 1
3. Populate other columns X_i:
  a. If unbiased, populate with p.
  b. If biased, populate with p dependent on Y^
4. Duplicate dataset with counterfactuals
  a. Replace A with unprotected attribute 0
  b. Flip outcome Y^ with probability eps
========================
OUTPUT
dataset   : (m + 2) x n sized numpy matrix, where number of columns
is the number of m additional attributes plus the protected attribute
and the outcome. [X_0, ..., X_m, A, Y^]
'''

def generate_dataset_values(m, n, biased, eps, p):
  validate_args(m, biased, p)

  def generate_unbiased_attribute_column(n):
    return (np.random.random(n) < p).astype(np.int64)
    
  def generate_biased_attribute_column(y, n):
    # Define probability of attribute x given outcome y (from write-up)
    p_x_y = p
    p_x_ny = 1 - p
    # Fill in attribute x based on outcome y
    y_idx = y == 1
    y_size = (np.sum(y)).astype(np.int64)
    x = np.full(n, False)
    x[y_idx] = np.random.random(y_size) < p_x_y
    x[np.logical_not(y_idx)] = np.random.random(n - y_size) < p_x_ny
    return x.astype(np.int64)

  # Step 1: Populate outcome
  y = np.concatenate([np.ones(n//4), np.zeros(n//2 - n//4)])
  
  # Step 2: Populate protected attribute
  a = np.ones(n//2)

  # Step 3: Populate additional attribute columns
  if biased:
    x_columns = np.vstack([generate_biased_attribute_column(y, n//2) for i in range(m)])
  else:
    x_columns = np.vstack([generate_unbiased_attribute_column(n//2) for i in range(m)])

  d_a = np.vstack([x_columns, a, y]).T

  # Step 4: Calculate counterfactuals
  na = np.zeros(n//2)
  y_na = np.copy(y)
  flip_idx = np.random.random(n//2) < eps
  y_na[flip_idx] = 1 - y_na[flip_idx]
  d_na = np.vstack([x_columns, na, y_na]).T

  return np.vstack([d_a, d_na])

def validate_args(m, biased, p):
  if m < 1:
    raise ValueError("m must be greater than 0.")

  if not 0.0 <= p <= 1.0:
    raise ValueError("Value p must be between 0.0 and 1.0.")


def generate_dataset(m, n, biased, eps, p):
  columns = ['X{}'.format(str(i)) for i in range(m)] + ['A', 'O']
  values = generate_dataset_values(m, n, biased, eps, p)
  return pd.DataFrame(data=values, columns=columns)

def validate_dataset(dataset):
  m = len(dataset.columns) - 2
  n = len(dataset.index)
  a = dataset.A.value_counts()[1]
  a_prime = dataset.A.value_counts()[0]

  o = dataset.groupby(['A', 'O']).size()[1][1]
  o_prime = dataset.groupby(['A', 'O']).size()[0][1]

  p_y_a = float(o) / a
  p_y_na = float(o_prime) / float(a_prime)

  eps = abs(p_y_a - p_y_na)

  p_biased = dataset.groupby(['X0', 'O']).size()[1][1] / dataset.X0.value_counts()[1]
  p_unbiased = dataset.X0.value_counts()[1] / n

  a_corr = dataset['O'].corr(dataset['A'])
  x_corr = dataset['O'].corr(dataset['X0'])

  indices = [list(i) for i in it.product([0, 1], repeat=m)]
  df = dataset.groupby(list(dataset.columns.values)).size()
  delta = 0
  for index_set in indices:
    cut = None
    for i, j in enumerate(index_set):
      if i == 0:
        cut = df[j]
      else:
        cut = cut[j]

    prob_pos = 0
    prob_neg = 0

    if 1 in cut:
      pos_a = cut[1]
      prob_pos = pos_a[1] / pos_a.sum() if 1 in pos_a else 0

    if 0 in cut:
      neg_a = cut[0]
      prob_neg = neg_a[1] / neg_a.sum() if 1 in neg_a else 0

    delta = delta + abs(prob_pos - prob_neg) * (cut.sum() / n)

  return m, n, delta, eps



