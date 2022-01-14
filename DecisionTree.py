import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle

url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic=pd.read_csv(url)
print(titanic.columns)
print(titanic.head())
print(titanic.isnull().sum())

titanic['Age'].fillna(value=titanic['Age'].mean(),inplace=True)
x=titanic['Cabin'].mode()[0]
titanic['Cabin'].fillna(x,inplace=True)

x=titanic['Embarked'].mode()[0]
titanic['Embarked'].fillna(x,inplace=True)

titanic.drop(['PassengerId','Name'],axis=1,inplace=True)
y=titanic['Survived']
X=titanic.drop(columns='Survived',axis=1)
y1=X.loc[:,'Sex']
y1=LabelEncoder().fit_transform(y1)
X['Sex']=y1
X=X.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare']]
X=X.astype(float)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25, random_state=234)
dt=DecisionTreeClassifier(criterion='entropy',max_depth=2)
dt.fit(xtrain,ytrain)
pre_ytest=dt.predict(xtest)

import numpy as np
narr=np.array([3,0,22,1,0,7.2500])
narr=narr.reshape(1,-1)
predy=dt.predict(narr)
#predy=dt.predict([1,0,22,3,2,34.45])
print(predy)

accuracy=accuracy_score(ytest,pre_ytest)
print(accuracy)
cm=confusion_matrix(ytest,pre_ytest)

print(cm)
cr=classification_report(ytest,pre_ytest)
print(cr)
feature_names=X.columns
class_names=[str(colname) for colname in dt.classes_]

import graphviz
from sklearn.tree import plot_tree
fig=plt.figure(figsize=(10,10))
plot_tree(dt,feature_names=feature_names,class_names=class_names,filled=True)
plt.savefig('Titanic Decision Tree')
filenm='MLAssign_DecisionTree.pickle'
pickle.dump(dt,open(filenm,'wb'))

