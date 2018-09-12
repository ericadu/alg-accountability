import numpy as np

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
    # Define probability of outcome y using marginal probabilities based on attribute
    p_y = p_a * p_y_a + (1 - p_a) * p_y_na
    # Define probability of attribute x given outcome y (from write-up)
    p_x_y = (p * (p_y + p - 1)) / (p_y * (2 * p - 1))
    p_x_ny = ((1 - p) * (p_y + p - 1)) / ((1 - p_y) * (2 * p - 1))
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
    x_columns = np.vstack([generate_biased_attribute_column(y) for i in range(m)])

  return np.vstack([x_columns, a, y]).T


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
