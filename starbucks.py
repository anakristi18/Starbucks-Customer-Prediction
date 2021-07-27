import pandas as pd
import numpy as np

#read data file 
starbucks = pd.read_csv('../input/starbucks-customer-retention-malaysia-survey/Starbucks satisfactory survey encode cleaned.csv')
starbucks.head()

starbucks.info()

#drop variable with index 10-15
starbucks.drop(starbucks.iloc[:,10:16], axis=1,inplace=True)
starbucks

#drop variable with index 18-26
starbucks.drop(starbucks.iloc[:,18:26], axis=1,inplace=True)
starbucks

starbucks.info()

#check missing value
starbucks.isnull().sum()

#delete variable 'id'
del starbucks['Id']

#import package seaborn
import seaborn as sns

#change categorical variable to string
starbucks['method']=starbucks['method'].map({0:'dine in', 1:'drive thru', 2:'take away', 3:'never', 4:'others'})
starbucks['status']=starbucks['status'].map({0:'student', 1:'self employed', 2:'employed', 3:'house wife'})

#count value from variable membershipCard
member0 = starbucks[starbucks.membershipCard==0].status.value_counts()
member1 = starbucks[starbucks.membershipCard==1].status.value_counts()

#import package matplotlib
import matplotlib.pyplot as plt

#change catgeorical variable membershipCard to string
starbucks['membershipCard']=starbucks['membershipCard'].map({0:'Yes', 1:'No'})

#create barchat form variable method
sns.countplot(starbucks.method, hue=starbucks.membershipCard)

#count value from status variable
starbucks['status'].value_counts()

#create pie chart
plt.figure(figsize=(5,5))
area = [58,37,16,2]
labels = ['employed','student','self employed','house wife']
colors = ['#7CB342','#C0CA33','#FFB300','#F57C00']
plt.pie(area,labels=labels,colors=colors,startangle=45,autopct='%1.1f%%',shadow=True,explode=(0,0,0,0.1))
plt.show()  

#create histogram
starbucks['income']=starbucks['income'].map({0:'< Rp 85.986,64', 1:'Rp 85.986,64 - Rp 171.973,68', 2:'Rp 171.973,68 - Rp 343.947,37', 3:'Rp 343.947,37 - Rp 515.921,05', 4:'> Rp 515.921,05'})
x = starbucks['income'].value_counts()
hist = x.hist(bins=5)
hist

starbucks.info()

#change categorical variable into numerik
starbucks['method']=starbucks['method'].map({'dine in':0,'drive thru':1,'take away':2,'never':3,'others':4})
starbucks['status']=starbucks['status'].map({'student':0,'self employed':1,'employed':2,'house wife':3})
starbucks['membershipCard']=starbucks['membershipCard'].map({'Yes':0,'No':1})
starbucks['income']=starbucks['income'].map({'< Rp 85.986,64':0,'Rp 85.986,64 - Rp 171.973,68':1,'Rp 171.973,68 - Rp 343.947,37':2,'Rp 343.947,37 - Rp 515.921,05':3,'> Rp 515.921,05':4})

starbucks.info()

starbucks['method']=starbucks['method'].fillna(0).astype(np.int64)

#create variable x
x = starbucks.iloc[:,0:17]

#create variable y
y = starbucks.iloc[:,17]

from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer,recall_score
from sklearn.metrics import roc_curve, auc
from numpy import mean

#feature selection using RFE

#defining model to build
model = DecisionTreeClassifier(random_state=123, criterion='entropy')
#create the RFE model and select 7 attributes
rfe = RFE(model,7,step=1)
x_new = rfe.fit_transform(x,y)
x_dt = pd.DataFrame(x_new)
x_dt.columns = ['gender','status','income','membershipCard','priceRate','promoRate','serviceRate']
print(x_new.shape)
print(rfe.get_support())

#define train data and test data
from sklearn.model_selection import train_test_split
#set 80% training and 20% testing
x_train,x_test,y_train,y_test=train_test_split(x_new,y,test_size=0.2,random_state=123)
model.fit(x_train,y_train)
accuracy = cross_val_score(model,x_train,y_train,cv=7)
print(accuracy.mean())
sensitivity = make_scorer(recall_score,pos_label=1)
Sensitivity = cross_val_score(model,x_train,y_train,cv=7,scoring=sensitivity)
print(Sensitivity.mean())
spesificity = make_scorer(recall_score,pos_label=0)
Spesificity = cross_val_score(model,x_train,y_train,cv=7,scoring=spesificity)
print(Spesificity.mean())

from sklearn.metrics import confusion_matrix
model.score(x_train,y_train)
y_pred = model.predict(x_test)
print(confusion_matrix(y_test,y_pred))

import graphviz
from sklearn.tree import export_graphviz
tree = export_graphviz(model,out_file='tree.dot',feature_names=x_dt.columns,class_names=['loyal','disloyal'],filled=True,rounded=True)


