'''
Dataset generator satisfying epsilon statistical parity. All dataset values are binary.
A base dataset has 1 protected attribute column and an output.
========================
INPUTS
m       : int     [1, )   :   number of additional attributes
n       : int     [1, )   :   number of samples
biased  : boolean         :   unbiased or biased columns
eps     : float   [0, 1]  :   epsilon
p_y_a   : float   [0, 1]  :   positive outcome likelihood
p_y_na  : float   [0, 1]  :   negative outcome likelihood
p_a     : float   [0, 1]  :   attribute likelihood
p       : float   [0, 1]  :   probability of outcome given unprotected attribute
========================
ALGORITHM
1. Populate protected attribute A with p_a
2. Populate outcome y using attribute A and probabilities p_y_a, p_y_na
3. Populate f0:
  a. If unbiased, populate with p = p_y_a - eps/2
  b. If biased, we use special case of Pr[Y = 1 | f_0 = 0] and Pr[Y = 1 | f_0 = 1]
========================
OUTPUT
dataset   : (m + 2) x n sized numpy matrix, where number of columns
is the number of m additional attributes plus the protected attribute
and the outcome.
'''

def generate_dataset(m, n, biased, eps, p_y_a, p_y_na, p_a, p):
  dataset = np.zeros((num_samples, num_columns + 2))
  count = 0
  for i in range(n):
    dataset[i, :] = generate_row(m, biased, eps, p_y_a, p_y_na, p_a, p)
  return dataset

def generate_row(m, biased, eps, p_y_a, p_y_na, p_a, p):
  protected_attribute = 1 if random() < p_a else 0

  outcome_if_a = protected_attribute == 1 and random() < p_y_a
  outcome_if_na = protected_attribute == 0 and random () < p_y_na

  outcome = 1 if outcome_if_a or outcome_if_na else 0

  if biased:
    pos_outcome_likelihood = p_a * p_y_a + (1 - p_a) * p_y_na
    attribute_absent = (pos_outcome_likelihood - 1 + p) / (2 * p - 1)
    attributes = [0 if random() < attribute_absent else 1 for _ in range(m)]
  else:
    attributes = [1 if random() < p else 0 for _ in range(m)]

  return [outcome, attribute] + attributes
