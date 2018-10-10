import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

'''
Model generator producing datasets that satisfy epsilon statistical parity. 
========================
INPUTS: DATASET
mu      : float    ( , )    :   mean of distribution
sigma   : float             :   std. deviation of distribution
p_a     : float    [0, 1]   :   Pr[ A = 1 ]

INPUTS: MODEL
b0      : float    ( , )    :   intercept 
b1      : float    ( , )    :   coefficent of X
b2      : float    ( , )    :   coefficient of A
========================
ALGORITHM
Dataset Generation
1. Populate A column independently with binary values 0 and 1, with probability p_a.
2. Independent of A, populate X column for both A = 0 and A = 1 with some Gaussian distribution.

Model Generation
1. Set coefficients.
2. Set intercept.
========================
OUTPUT
dataset   : 3 x n sized numpy matrix [X, A, O], where O is Y^
'''

def gaussian(mu, sigma, n):
  return np.random.normal(mu, sigma, n)

def binary(p_a, n):
    return (np.random.random(n) < p_a).astype(np.int64)

def generate_attributes(mu_a, sigma_a, mu_na, sigma_na, p_a, n):
  n_a = int(p_a * n)
  X_a = gaussian(mu_a, sigma_a, n_a)
  X_na = gaussian(mu_na, sigma_na, n - n_a)
  a = np.ones(n_a)
  na = np.zeros(n - n_a)
  X = np.concatenate([X_a, X_na])
  A = np.concatenate([a, na])
  return np.vstack([X, A]).T

def outcome(attributes, mu):
  X = attributes[:,0]
  Y = (X > mu).astype(np.int64)
  return Y

def generate_predicted_dataset(mu_a, sigma_a, mu_na, sigma_na, p_a, n, clf):
  columns = ['X', 'A', 'O']
  attributes = generate_attributes(mu_a, sigma_a, mu_na, sigma_na, p_a, n)
  O = clf.predict(attributes)
  values = np.vstack([attributes.T, O]).T
  return pd.DataFrame(data=values, columns=columns)

def generate_model(b0, b1, b2):
  clf = LogisticRegression(penalty='l2', C=0.01)
  clf.coef_ = np.array([[b1, b2]])
  clf.intercept_ = np.array([b0])
  clf.classes_ = np.array([0, 1])
  return clf

def validate_dataset(df):
  n = df.A.size
  a = df.A.value_counts()[1]
  a_prime = n - a
  p_a = float(a) / n
  size = df.groupby(['A', 'O']).size()

  o = size[1][1] if 1 in size[1] else 0
  o_prime = size[0][1] if 1 in size[0] else 0
  p_y_a = float(o) / a
  p_y_na = float(o_prime) / float(a_prime)
  eps = p_y_a - p_y_na
  sigma_na, sigma_a = df.groupby('A').X.std()
  mu_na, mu_a = df.groupby('A').X.mean()
  results = [mu_na, mu_a, sigma_na, sigma_a, p_a, eps]
  return ','.join([str(round(i, 2)) for i in results])
