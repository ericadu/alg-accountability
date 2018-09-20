import numpy as np
import pandas as pd

'''
Dataset generator satisfying epsilon statistical parity. All dataset values are binary.
A base dataset has 1 protected attribute column and an output.
========================
INPUTS
m       : int     [1, )   :   number of additional attributes
n       : int     [1, )   :   number of samples
biased  : boolean         :   unbiased or biased columns
eps     : float   [0, 1]  :   | Pr[ Y^ = 1 | A = 1 ] - Pr[ Y^ = 1 | A = 0 ] | < epsilon 
p_y_A   : float   [0, 1]  :   Pr[ Y^ = 1 | A ]
p_a     : float   [0, 1]  :   Pr[ A = 1] attribute likelihood
p       : float   [0, 1]  :   Pr[ X_i = 1 | Y^ = 1] (biased) or Pr[X_i = 1] (unbiased) 
========================
ALGORITHM
1. Populate protected attribute A with p_a
2. Populate outcome y using attribute A and probability p_y_a and p_y_na calculated from eps and p_y_A
3. Populate other columns X_i:
  a. If unbiased, populate with p.
  b. If biased, populate with p dependent on Y^
========================
OUTPUT
dataset   : (m + 2) x n sized numpy matrix, where number of columns
is the number of m additional attributes plus the protected attribute
and the outcome. [X_0, ..., X_m, A, Y^]
'''

def generate_dataset_values(m, n, biased, eps, p_y_A, p_a, p):
  p_y_a = p_y_A + eps/2
  p_y_na = p_y_A - eps/2
  validate_args(m, biased, p_y_a, p_y_na, p_a, p)

  def generate_protected_attribute_column():
    # Can also convert to int8 since just 0/1, but need to be careful
    # how it is used downstream. Don't want to accidentally cause overflow
    # issues in FairML or FairTest
    return (np.random.random(n) < p_a).astype(np.int64)

  def generate_outcome_column(attr):
    attr_idx = attr == 1
    attr_size = np.sum(attr)
    y = np.full(n, False)
    # Fill in outcomes based on attribute
    y[attr_idx] = np.random.random(attr_size) < p_y_a
    y[np.logical_not(attr_idx)] = np.random.random(n - attr_size) < p_y_na
    return y.astype(np.int64)

  def generate_unbiased_attribute_column():
    return (np.random.random(n) < p).astype(np.int64)
    
  def generate_biased_attribute_column(y):
    # Define probability of attribute x given outcome y (from write-up)
    p_x_y = p
    p_x_ny = 1 - p
    # Fill in attribute x based on outcome y
    y_idx = y == 1
    y_size = np.sum(y)
    x = np.full(n, False)
    x[y_idx] = np.random.random(y_size) < p_x_y
    x[np.logical_not(y_idx)] = np.random.random(n - y_size) < p_x_ny
    return x.astype(np.int64)

  # Step 1: Populate protected attribute
  a = generate_protected_attribute_column()

  # Step 2: Populate outcome
  y = generate_outcome_column(a) 

  # Step 3: Populate additional attribute columns
  if biased:
    x_columns = np.vstack([generate_biased_attribute_column(y) for i in range(m)])
  else:
    x_columns = np.vstack([generate_unbiased_attribute_column(y) for i in range(m)])

  return np.vstack([x_columns, a, y]).T

def validate_args(m, biased, p_y_a, p_y_na, p_a, p):
  if m < 1:
    raise ValueError("m must be greater than 0.")

  floats = {
    'p_y_a': p_y_a,
    'p_y_na': p_y_na,
    'p_a': p_a,
    'p': p,
  }

  for name, val in floats.items():
    if not 0.0 <= val <= 1.0:
      formatted_val = str(val)
      raise ValueError("Value {} must be between 0.0 and 1.0. Currently equal to: {}".format(name, formatted_val))


def generate_dataset(m, n, biased, eps, p_y_A, p_a, p):
  columns = ['X{}'.format(str(i)) for i in range(m)] + ['A', 'O']
  values = generate_dataset_values(m, n, biased, eps, p_y_A, p_a, p)
  return pd.DataFrame(data=values, columns=columns)

def validate_dataset(dataset):
  m = len(dataset.columns) - 2
  n = len(dataset.index)
  a = dataset.A.value_counts()[1]
  a_prime = dataset.A.value_counts()[0]

  p_a = float(a) / n

  o = dataset.groupby(['A', 'O']).size()[1][1]
  o_prime = dataset.groupby(['A', 'O']).size()[0][1]

  p_y_a = float(o) / a
  p_y_na = float(o_prime) / float(a_prime)
  p_y_A = (p_y_a + p_y_na) / 2
  eps = p_y_a - p_y_na

  p_biased = dataset.groupby(['X0', 'O']).size()[1][1] / dataset.X0.value_counts()[1]
  p_unbiased = dataset.X0.value_counts()[1] / n

  a_corr = dataset['O'].corr(dataset['A'])
  x_corr = dataset['O'].corr(dataset['X0'])

  return m, n, eps, p_y_A, p_a, p_biased, p_unbiased, a_corr, x_corr
