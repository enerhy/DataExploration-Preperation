# Pandas

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\Machine Learning\zero_to_deep_learning_video\data/titanic-train.csv')

df.head()
df.describe()
df.isnull().any() / df.shape()

#Indexing
row3 = df.iloc[3] #calls rows or columns by index
df.loc[0:4, 'Ticket'] #calls rows or columns by column name
df['Ticket'].head()
df[['Ticket', 'Embarked']].head() # note we have here 2 brackets

#Selections
cond1 = df['Age'] > 60 #returns true or falls values for the column
cond2 = df[df['Age'] > 60] #returns the DF with the statisfied condition
cond3 = df.query("Age > 60") #same as previous one
cond4 = df[(df['Age'] == 11) & (df['SibSp'] == 5)] #AND condition
cond5 = df[(df.Age == 11) & (df.SibSp == 5)] #same as previous
cond6 = df[(df.Age == 11) | (df.SibSp == 5)] #OR
cond7 = df.query('(Age == 11) | (SibSp == 5)') #same as previous

cond8 = df['Embarked'].unique() #shows the unique values
print(cond8)

#Aggregation
ag1 = df['Survived'].value_counts() #aggregates for the values
ag1 = df['Survived'].value_counts().plot(kind = 'bar')

ag2 = df['Pclass'].value_counts()
# if we want the percentage: df.Pclass.value_counts() / df.Pclass.notnull().sum()
ag3 = df.groupby(['Pclass', 'Survived'])['PassengerId'].count() #Group By

df['Age'].min()
df['Age'].max()
df['Age'].mean()
df['Age'].median()

ag4 = df.groupby('Survived')['Age'].mean()
ag5 = df.groupby('Survived')['Age'].std()
mean_age_by_survived = ag4
std_age_by_survived = ag5
ag6 = df.groupby(['Age', 'Survived'])['PassengerId'].count()


df1 = mean_age_by_survived.reset_index()
#reset index makes it a DataFrame from a serie
df1_rounded = mean_age_by_survived.round(0).reset_index()
df2 = std_age_by_survived.round(0).reset_index()

#Merge
df3 = pd.merge(df1, df2, on='Survived')
#Renaming the columns - the order is important
df3.columns = ['Survived', 'Average Age', 'Age Standard Deviation']

#Pivot Tables
pivot = df.pivot_table(index='Pclass',
                       columns='Survived',
                       values='PassengerId',
                       aggfunc='count')
pivot = pivot.reset_index()

#Sorting
df.sort_values('Age', ascending = False).head

#Correlations
df['IsFemale'] = df['Sex'] == 'female' #Adding a column in the DF at the end
#in this case is a binary variable in true or falls
correlated_with_survived = df.corr()['Survived'].sort_values()
correlated_with_survived


'---------Data Split with Pandas-------------'
from pandas.tseries.offsets import MonthEnd

df['Time_coulmn'] = pd.to_datetime(df['Time_coulmn']) + MonthEnd(1)  #MonthEnd takes the Month value and makes a date with the last day
df = df.set_index('Time_coulmn')

split_date = pd.Timestamp('01-01-2011')   #Defining the date to split

train = df.loc[:split_date, ['Column_to_split']]






'-----Visual Exploration------'

import seaborn as sns
sns.pairplot(df, hue='class') #hue - the feature/ column name








