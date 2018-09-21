import numpy as np
import pandas as pd

'''
Dataset generator satisfying delta statistical parity and epsilon counterfactual fairness.
All dataset values are binary.
========================
INPUTS
m       : int     [1, )   :   number of additional attributes
n       : int     [1, )   :   number of samples
biased  : boolean         :   unbiased or biased columns
delta     : float   [0, 0.5]  :   | Pr[ Y^ = 1 | A = 1 ] - Pr[ Y^ = 1 | A = 0 ] | < delta
eps   : float   [0, 0.5]  :   | Pr[ Y^ = 1 | A = 1, X ] - Pr[ Y^ = 1 | A = 0, X ] | < eps
p       : float   [0, 1]  :   Pr[ X_i = 1 | Y^ = 1] (biased) or Pr[X_i = 1] (unbiased) 
========================
ALGORITHM
1. Populate outcome Y^ with even split
2. Populate protected attribute A with protected attribute 1
3. Populate other columns X_i:
  a. If unbiased, populate with p.
  b. If biased, populate with p dependent on Y^
4. Duplicate dataset with counterfactuals
  a. Replace A with unprotected attribute 0
  b. Flip outcome Y^ with probability eps
  c. Remove some rows of Y^=0, A=0 to achieve statistical parity
========================
OUTPUT
dataset   : (m + 2) x n sized numpy matrix, where number of columns
is the number of m additional attributes plus the protected attribute
and the outcome. [X_0, ..., X_m, A, Y^]
'''

def generate_dataset_values(m, n, biased, eps, delta, p):
  validate_args(m, biased, p)

  def generate_unbiased_attribute_column(n):
    return (np.random.random(n) < p).astype(np.int64)
    
  def generate_biased_attribute_column(y, n):
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

  # Step 0: Calculate divisions in order to build dataset size n with statistical parity
  # TODO(nirvan): See notebook for calculations
  # Divide dataset into 4 sets of n/k
  k = (4 * delta + 4.0) / (1.0 + 2 * delta)
  split_size = int(n // k)
  # Remove l values of one set in order to achieve delta statistical parity
  l = int((delta * n) // (1.0 + delta))
  if l > split_size:
    raise ValueError('Epsilon value for statistical parity unachievable')

  # Step 1: Populate outcome
  y = np.concatenate([np.ones(split_size), np.zeros(split_size)])
  
  # Step 2: Populate protected attribute
  a = np.ones(2 * split_size)

  # Step 3: Populate additional attribute columns
  if biased:
    x_columns = np.vstack([generate_biased_attribute_column(y, 2*split_size) for i in range(m)])
  else:
    x_columns = np.vstack([generate_unbiased_attribute_column(2*split_size) for i in range(m)])

  d_a = np.vstack([x_columns, a, y]).T

  # Step 4: Calculate counterfactuals and remove l rows
  y_na = np.copy(y)
  flip_idx = np.random.random(2*split_size) < eps
  y_na[flip_idx] = 1 - y_na[flip_idx]

  print(split_size)
  print(np.where(y_na == 0)[0].shape[0])
  if np.where(y_na == 0)[0].shape[0] > split_size:
    na_idx = np.concatenate([
        np.where(y_na == 1)[0],
        np.where(y_na == 0)[0][:-l]
    ])
  else:
    na_idx = np.concatenate([
        np.where(y_na == 1)[0][l:],
        np.where(y_na == 0)[0]
    ])

  y_na = y_na[na_idx]
  na = np.zeros(2*split_size - l)
  x_na = x_columns[:, na_idx]
  d_na = np.vstack([x_na, na, y_na]).T

  return np.vstack([d_a, d_na])

def validate_args(m, biased, p):
  if m < 1:
    raise ValueError("m must be greater than 0.")

  if not 0.0 <= p <= 1.0:
    raise ValueError("Value p must be between 0.0 and 1.0.")


def generate_dataset(m, n, biased, eps, delta, p):
  columns = ['X{}'.format(str(i)) for i in range(m)] + ['A', 'O']
  values = generate_dataset_values(m, n, biased, eps, delta,  p)
  return pd.DataFrame(data=values, columns=columns)

def validate_dataset(dataset):
  raise NotImplementedError()
