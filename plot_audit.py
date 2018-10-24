import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def round_array(array):
  return [round(i, 3) for i in array]

def plot_heatmap():
df1 = pd.read_csv('output/drop_white_low_risk/propublica.csv')
df2 = pd.read_csv('output/drop_black_high_risk/propublica.csv')
df3 = pd.read_csv('output/propublica.csv')
df1_mean = round_array(df1.groupby('delta').epsilon.describe()['mean'].values)
df2_mean = round_array(df2.groupby('delta').epsilon.describe()['mean'].values)
df3_mean = round_array(df3.groupby('delta').epsilon.describe()['mean'].values)

data = np.stack((df1_mean, df2_mean, df3_mean))
x_labels = list(df1.delta.unique())
y_labels = ['Low//Caucasian', 'High//African-American', 'Both']
y_dim, x_dim = data.shape
plt.imshow(data, cmap='Blues', vmin=0, vmax=0.4)
plt.xticks(np.arange(x_dim), x_labels)
plt.yticks(np.arange(y_dim), y_labels)
plt.tick_params(labeltop=True, labelbottom=False, length=0)
for j in range(y_dim):
  for i in range(x_dim):
    plt.text(i, j, '{0:.2f}'.format(data[j, i]), ha='center', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('heatmap-1.png')

if __name__ == '__main__':
  df = pd.read_csv('output/propublica.csv')
  # pl = df.boxplot(column='epsilon', by='delta')
  pl = df.plot.scatter(x='delta', y='epsilon')
  fig = pl.get_figure()
  fig.savefig("output/propublica.png")