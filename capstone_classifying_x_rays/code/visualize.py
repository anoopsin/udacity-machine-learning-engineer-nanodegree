import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(sum(map(ord, 'categorical')))

df = pd.read_csv('Data_Entry_2017.csv')
df = df[['Finding Labels', 'Patient Age', 'Patient Gender']]

g = sns.countplot(x='Patient Age', hue='Patient Gender', data=df[df['Finding Labels'].str.contains('Cardiomegaly')], orient='v')

x = np.arange(0,100,10)
g.set_xlim(0,90)
g.set_xticks(x)
g.set_xticklabels(x)

g.get_figure().savefig("output.png")

