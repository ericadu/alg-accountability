import numpy as np
from random import random

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
p       : float   [0, 1]  :   Pr[ Y^ = 1 | f0 = 0] prob. of outcome given unprotected attribute
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

def generate_dataset(m, n, biased, eps, p_y_A, p_a, p):
  p_y_a = p_y_A + eps/2
  p_y_na = p_y_A - eps/2
  validate_args(m, biased, p_y_a, p_y_na, p_a, p)

  def get_outcome(x):
    if (x == 1 and random() < p_y_a) or (x == 0 and random() < p_y_na):
      return 1
    else:
      return 0

  def get_attr(y):
    p_y_eq_1 = p_a * p_y_a + (1 - p_a) * p_y_na

    p_attr_0 = (p_y_eq_1 - 1 + p) / (2 * p - 1)

    p_attr_0_given_y_eq_1 = (p * p_attr_0) / p_y_eq_1
    p_attr_0_given_y_eq_0 = (1 - p) * p_attr_0 / (1 - p_y_eq_1)

    attr_if_1 = y == 1 and (random() < p_attr_0_given_y_eq_1)
    attr_if_0 = y == 0 and (random() < p_attr_0_given_y_eq_0)

    if attr_if_1 or attr_if_0:
      return 0
    else:
      return 1

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


def validate_args(m, biased, p_y_a, p_y_na, p_a, p):
  if m < 1:
    raise ValueError("m must be greater than 0.")

  p_y_eq_1 = p_a * p_y_a + (1 - p_a) * p_y_na
  p_attr_0 = (p_y_eq_1 - 1 + p) / (2 * p - 1) if biased else p
  p_attr_0_given_y_eq_1 = (p * p_attr_0) / p_y_eq_1
  p_attr_0_given_y_eq_0 = (1 - p) * p_attr_0 / (1 - p_y_eq_1)

  floats = {
    'p_y_a': p_y_a,
    'p_y_na': p_y_na,
    'p_a': p_a,
    'p': p,
    'p_y_eq_1': p_y_eq_1,
    'p_attr_0': p_attr_0,
    'p_attr_0_given_y_eq_1': p_attr_0_given_y_eq_1,
    'p_attr_0_given_y_eq_0': p_attr_0_given_y_eq_0 
  }

  for name, val in floats.items():
    if not 0.0 <= val <= 1.0:
      raise ValueError("{} must be between 0.0 and 1.0. Currently equal to: ".format(name, str(val)))