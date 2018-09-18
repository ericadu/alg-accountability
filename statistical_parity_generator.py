import numpy as np
import pandas as pd
from random import random

'''
Dataset generator satisfying epsilon statistical parity. All dataset values are binary.
A base dataset has 1 protected attribute column and an output.
========================
INPUTS
exp     : str             :   experiment name
m       : int     [1, )   :   number of additional attributes
n       : int     [1, )   :   number of samples
biased  : boolean         :   unbiased or biased columns
eps     : float   [0, 1]  :   | Pr[ Y^ = 1 | A = 1 ] - Pr[ Y^ = 1 | A = 0 ] | < epsilon 
p_y_A   : float   [0, 1]  :   Pr[ Y^ = 1 | A ]
p_a     : float   [0, 1]  :   Pr[ A = 1] attribute likelihood
p       : float   [0, 1]  :   Pr[ Y^ = 1 | f0 = 1] prob. of outcome given unprotected attribute
========================
ALGORITHM
1. Populate protected attribute A with p_a
2. Populate outcome y using attribute A and probability p_y_a and p_y_a' calculated from eps and p_y_A
3. Populate f0:
  a. If unbiased, populate with p.
  b. If biased, we use special case of Pr[Y = 1 | f_0 = 0] and Pr[Y = 1 | f_0 = 1]
========================
OUTPUT
dataset   : (m + 2) x n sized numpy matrix, where number of columns
is the number of m additional attributes plus the protected attribute
and the outcome. columns are f0, ..., f_(m-1), protected_attr, outcome.
'''

def generate_dataset_values(exp, m, n, biased, eps, p_y_A, p_a, p):
  p_y_a = p_y_A + eps/2
  p_y_na = p_y_A - eps/2
  validate_args(exp, m, biased, p_y_a, p_y_na, p_a, p)

  def get_outcome(x):
    if (x == 1 and random() < p_y_a) or (x == 0 and random() < p_y_na):
      return 1
    else:
      return 0

  def get_attr(y):
    if (y == 1 and random() < p) or (y == 0 and random() < 1 - p):
      return 1
    else:
      return 0

  v_outcome_func = np.vectorize(get_outcome)
  v_attr_func = np.vectorize(get_attr)

  # Step 1: Populate protected
  protected_attr = np.array([[1 if random() < p_a else 0 for _ in range(n)]])

  # Step 2: Populate outcome
  outcome = v_outcome_func(protected_attr)

  # Step 3: Populate columns
  columns = np.zeros((m, n))
  for i in range(m):
    if biased:
      columns[i,:] = v_attr_func(outcome)
    else:
      columns[i,:]= [1 if random() < p else 0 for _ in range(n)]
  
  return np.concatenate((columns, protected_attr, outcome)).T
  

def validate_args(exp, m, biased, p_y_a, p_y_na, p_a, p):
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
      raise ValueError("In {}, value {} must be between 0.0 and 1.0. Currently equal to: ".format(exp, name, formatted_val))

def generate_dataset(exp, m, n, biased, eps, p_y_A, p_a, p):
  columns = ['X{}'.format(str(i)) for i in range(m)] + ['A', 'O']
  values = generate_dataset_values(exp, m, n, biased, eps, p_y_A, p_a, p)
  return pd.DataFrame(data=values, columns=columns)

def validate_dataset(dataset):
  a = dataset.A.value_counts()[1]
  a_prime = dataset.A.value_counts()[0]

  p_a = float(a) / len(dataset.index)

  o = dataset.groupby(['A', 'O']).size()[1][1]
  o_prime = dataset.groupby(['A', 'O']).size()[0][1]

  p_y_a = float(o) / a
  p_y_na = float(o_prime) / float(a_prime)
  p_y_A = (p_y_a + p_y_na) / 2
  eps = p_y_a - p_y_na

  p_biased = dataset.groupby(['X0', 'O']).size()[1][1] / dataset.X0.value_counts()[1]
  p_unbiased = dataset.X0.value_counts()[1] / len(dataset.index)


  return eps, p_y_A, p_a, p_biased, p_unbiased

