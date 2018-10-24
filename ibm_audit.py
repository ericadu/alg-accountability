import statistical_parity_generator as sp

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

'''
IBM audits for: 
- Consistency: how similar labels are (distance)
- Mean Difference: statisical parity
'''
def audit(m, n, biased, eps, p_y_A, p_a, p):
  dataset = sp.generate_dataset(m, n, biased, eps, p_y_A, p_a, p)
  bld = BinaryLabelDataset(df=dataset, label_names=['O'], protected_attribute_names=['A'], unprivileged_protected_attributes=[[1]], privileged_protected_attributes=[[0]])
  bldm = BinaryLabelDatasetMetric(bld, unprivileged_groups=[{'A': 1}], privileged_groups=[{'A': 0}])
  epsilon = bldm.statistical_parity_difference()
  print(epsilon)

if __name__ == '__main__':
  filename = '/Users/erica/Desktop/GitHub/data/compas_concat.csv'
  df = pd.read_csv(filename)
  size = df.index.size
  for delta in [0, 0.01, 0.05, 0.1, 0.15, 0.2]:
    remove_n = int(delta * size)
    a_df = df[df.race == 'African-American']
    c_df = df[df.race == 'Caucasian']

    drop_indices = np.random.choice(c_df.index, remove_n, replace=False)
    df_subset = c_df.drop(drop_indices)

    df_subset = pd.concat([df_subset, a_df])

    gb = df_subset.groupby(['race', 'score_text']).size()
    r_a = gb['African-American'].sum() if 'African-American' in gb else 0
    r_c = gb['Caucasian'].sum() if 'Caucasian' in gb else 0

    high_a = 0
    high_c = 0

    if r_a > 0:
      high_a = gb['African-American']['High'] / r_a
    
    if r_c > 0:
      high_c = gb['Caucasian']['High'] / r_c
    
    print("black: {}, white: {}, delta: {}, epsilon: {}".format(r_a, r_c, delta, abs(high_a - high_c)))
