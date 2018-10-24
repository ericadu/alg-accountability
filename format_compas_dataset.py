import tempfile
import tensorflow as tf
import numpy as np

'''
race: caucasian | african american | hispanic | native american | asian | other
          0             1               2             3             4       5

sex:  male    |   female
        0             1

age:  < 25    |   25-45   | 45+
        0            1        2

juv_fel_count: continuous integer
decile_score: 1 - 10, integer
juv_misd_count: continuous integer
juv_other_count: continuous integer
priors_count: continuous integer
c_charge_degree: F | M
'''

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = {**categorical_cols, **continuous_cols}
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(train_df)

def eval_input_fn():
  return input_fn(test_df)

filename = '/Users/erica/Desktop/GitHub/data/compas-scores-two-years.csv'
  # LABEL_COLUMN = 'two_year_recid'
LABEL_COLUMN = 'risk' # High risk = 1, Other = 0
CATEGORICAL_COLUMNS = ['sex', 'age_cat', 'race', 'c_charge_degree']
CONTINUOUS_COLUMNS = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']

df = pd.read_csv(filename)
df = df[df.days_b_screening_arrest <= 30]
df = df[df.days_b_screening_arrest >= -30]
filtered_df = df[['sex', 'age_cat', 'race', 'juv_fel_count', 'decile_score', 'score_text', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'is_recid', 'two_year_recid']]

msk = np.random.rand(len(df)) < 0.8
train_df = filtered_df[msk]
test_df = filtered_df[~msk]

gender = tf.contrib.layers.sparse_column_with_keys(column_name="sex", keys=["Female", "Male"])
age_cat = tf.contrib.layers.sparse_column_with_keys(column_name="age_cat", keys=['Greater than 45', '25 - 45', 'Less than 25'])
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=['African-American', 'Caucasian', 'Hispanic', 'Asian', 'Native American', 'Other'])
# decile_score = tf.contrib.layers.sparse_column_with_hash_bucket('decile_score', hash_bucket_size=10)
# score_text = tf.contrib.layers.sparse_column_with_hash_bucket('score_text', hash_bucket_size=3)
charge_degree = tf.contrib.layers.sparse_column_with_keys(column_name="c_charge_degree", keys=["F", "M"])

juv_fel_count = tf.contrib.layers.real_valued_column("juv_fel_count")
juv_misd_count = tf.contrib.layers.real_valued_column("juv_misd_count")
juv_other_count = tf.contrib.layers.real_valued_column("juv_other_count")
priors_count = tf.contrib.layers.real_valued_column("priors_count")

model_dir = '/Users/erica/Desktop/GitHub/model-0'

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

m = tf.contrib.learn.LinearClassifier(
  feature_columns=[gender,age_cat,race,charge_degree,juv_fel_count,juv_misd_count,juv_other_count,priors_count],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=model_dir
)
m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
  print("{}: {}".format(key, results[key]))
