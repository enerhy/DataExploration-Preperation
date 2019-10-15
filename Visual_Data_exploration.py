# # Visual Data Exploration with Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = np.random.normal(0, 0.1, 1000)
data2 = np.random.normal(1, 0.4, 1000) + np.linspace(0, 1, 1000)
data3 = 2 + np.random.random(1000) * np.linspace(1, 5, 1000)
data4 = np.random.normal(3, 0.2, 1000) + 0.3 * np.sin(np.linspace(0, 20, 1000))


#Stacking the data into an array
data = np.vstack([data1, data2, data3, data4]).transpose()

#Array into DataFrame using PANDAS specifying the column names
df = pd.DataFrame(data, columns=['data1', 'data2', 'data3', 'data4'])
df.head()

#LinePlot
plt.plot(df)
plt.title('LinePlot')
plt.legend(['data1', 'data2', 'data3', 'data4']) #its a list coresponding to 
                                                #the columns

plt.show()

# To make it a ScatterPlot
df.plot(style='.')

#Histogramm
plt.hist(data, bins = 50)
plt.title('Histogram')
plt.legend(['data1', 'data2', 'data3', 'data4']) #its a list coresponding to 

df.plot(kind='hist',
        bins=50,
        title='Histogram',
        alpha=0.6)


#Cummulative distribution
df.plot(kind='hist',
        bins=100,
        title='Cumulative distributions',
        normed=True,
        cumulative=True,
        alpha=0.4)

#BoxPlot - good to compare distributions with outliers and width
df.plot(kind='box',
        title='Boxplot')

#Subpolts. We furst create a grid and than we create
#the subplots on the specific place
fig, ax = plt.subplots(2, 2, figsize=(5, 5))

df.plot(ax=ax[0][0],
        title='Line plot')

df.plot(ax=ax[0][1],
        style='o',
        title='Scatter plot')

df.plot(ax=ax[1][0],
        kind='hist',
        bins=50,
        title='Histogram')

df.plot(ax=ax[1][1],
        kind='box',
        title='Boxplot')

plt.tight_layout()

#PyeChart
gt01 = df['data1'] > 0.1
piecounts = gt01.value_counts()

piecounts.plot(kind='pie',
               figsize=(5, 5),
               explode=[0, 0.15],
               labels=['<= 0.1', '> 0.1'],
               autopct='%1.1f%%',
               shadow=True,
               startangle=90,
               fontsize=16)

#HixBin Plot
data = np.vstack([np.random.normal((0, 0), 2, size=(1000, 2)),
                  np.random.normal((9, 9), 3, size=(2000, 2))])
df = pd.DataFrame(data, columns=['x', 'y'])

df.plot(kind='kde')
df.plot(kind='hexbin', x='x', y='y', bins=100, cmap='rainbow')

#PairPlot
#https://seaborn.pydata.org/generated/seaborn.pairplot.html
>>> import seaborn as sns; sns.set(style="ticks", color_codes=True)
>>> iris = sns.load_dataset("iris")
>>> g = sns.pairplot(iris)

g = sns.pairplot(iris, hue="species")

















