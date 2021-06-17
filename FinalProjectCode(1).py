# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 06:45:27 2020

@author: Garrett
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.tree import plot_tree


#Had some issues with the fit function, google suggested this version. Use if you get an error.
#pip install pandas==0.23.4


#Data taken from https://www.kaggle.com/spscientist/students-performance-in-exams
df = pd.read_csv("C:/Users/Garrett/Desktop/QMST 3339/StudentsPerformance.csv")


#Rename columns and drop null values.
df.columns = ['gender', 'race', 'parentDegree', 'lunch', 'prepCourse', 'mathScore', 'readingScore', 'writingScore'] 
df.dropna()


#Correlation of exam results. Reading and writing are closely correlated. Math is not extremely correlated with either reading or writing.
plt.figure(dpi=100)
plt.title('Correlation Analysis')
snb.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')
plt.xticks(rotation=60)
plt.yticks(rotation = 60)
plt.show()


#Adding a total average score per student.
df['total'] = (df['mathScore']+df['readingScore']+df['writingScore'])/3


#Six outliers in total average score.
IQR=np.nanpercentile(df['total'],75)-np.nanpercentile(df['total'],25)
upperBound=np.nanpercentile(df['total'],75)+1.5*IQR
lowerBound=np.nanpercentile(df['total'],25)-1.5*IQR
outliers=sum((df['total']>upperBound) | (df['total']<lowerBound))


#Scatterplot of math, reading, writing, and total average scores. 5 clusters.
testScores=df.iloc[:,5:]
kmeans = KMeans(n_clusters=5) 
scaler = MinMaxScaler()
scaled = scaler.fit_transform(testScores)
kmeans.fit(scaled)
(kmeans.cluster_centers_)

plt.scatter(scaled[:,0],scaled[:,1], c=kmeans.labels_, cmap='rainbow')


#Completing the prep course has a positive effect on scores.
course_gender = df.groupby(['gender','prepCourse']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='prepCourse', data=course_gender, kind='bar')
    

#If the parents have degrees, children have better test scores.
for i in range(len(df)):
    if df.iloc[i,2] in ['high school', 'some high school']:
        df.iloc[i,2] = 'no_Degree'
    else:
        df.iloc[i,2] = 'has_Degree'
        
course_gender = df.groupby(['gender','parentDegree']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='parentDegree', data=course_gender, kind='bar')


#Showing average grades by gender and race. Female, Group E performs the best. Male, Group A performs the worst. Females perform better overall on average.
race_gender = df.groupby(['gender','race']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='race', data=race_gender, kind='bar')


#Recoding the data numerically.
df[['gender']]=df[['gender']].replace('male', 1)
df[['gender']]=df[['gender']].replace('female', 2)

df[['race']]=df[['race']].replace('group A', 1)
df[['race']]=df[['race']].replace('group B', 2)
df[['race']]=df[['race']].replace('group C', 3)
df[['race']]=df[['race']].replace('group D', 4)
df[['race']]=df[['race']].replace('group E', 5)

df[['parentDegree']]=df[['parentDegree']].replace('no_Degree', 1)
df[['parentDegree']]=df[['parentDegree']].replace('has_Degree', 2)

df[['lunch']]=df[['lunch']].replace('standard', 1)
df[['lunch']]=df[['lunch']].replace('free/reduced', 2)

df[['prepCourse']]=df[['prepCourse']].replace('none', 1)
df[['prepCourse']]=df[['prepCourse']].replace('completed', 2)


#Failure will be less than 60. Passing will be equal or greater than 60.
df.loc[df.total < 60, 'outcome'] = 0
df.loc[df.total >= 60, 'outcome'] = 1


#Decision Tree
X = df[['gender','race', 'parentDegree', 'lunch', 'prepCourse']] 
Y = df[['outcome']] 
X_train, X_test, y_train, y_test = train_test_split(  
X, Y, test_size = 0.3, random_state = 100) 

clftitan_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 

clftitan_entropy.fit(X_train, y_train) 

y_pred = clftitan_entropy.predict(X_test)

feature_cols = ['gender','race', 'parentDegree', 'lunch', 'prepCourse']
plot_tree(clftitan_entropy,filled=True,rounded=True, 
feature_names = feature_cols,class_names=True, fontsize = 4.75)


#Accuracy 78%
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred)) 
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 
print("Report : ", classification_report(y_test, y_pred)) 

