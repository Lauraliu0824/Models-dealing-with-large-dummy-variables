import pandas as pd
import numpy as np
from random import randnt
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn import svm
import matplotlib.pyplot as plt

#Create dummy variables
def creatDummy(df,variable):   
    colnames = list(df.columns.values)    
	for elem in df[variable].unique():  
	    name = str(elem)  
		if name in colnames:
		   name += str(randint(0,1000))
		   print(name)
		   
		df[str(name)] = df[variable] == elem   
		
    df = df.drop([str(df[variable].unique()[0]),variable], axis=1)
	print(df.head())
	return df

	
LIST = ['PRODUCT_TYPE','SM_LOAN_STATUS','MULTIFEE_RATE']

for v in LIST:
    df =  creatDummy(df,v)    
	
	
	
#PCA
from sklearn.decomposition import PCA
msk = np.random.rand(len(df)) < 0.8
x = trainingdata = df[msk]
X = testdata = df[~msk]

y = x['PFLAG'].values
x = x.drop(['PFLAG'],axis =1).values
Y = X['PFLAG'].values
X = X.drop(['PFLAG'], axis =1).values

pca = PCA(n_components = 50)
x= pca.fit_transform(x)
X= pca.transform(X)

Print(x.shape)

#modeling
clf = RandomForestClassifier(n_estimators = 400, max_depth = 4)
clf = LogisticRegression(random_state = 0, solver = 'lbfgs')
clf = svm.svc(kernel = 'linear')  #rbf, linear, ploy, sigmoid
clf = AdaBoostClassfier(DecisionTreeClassifier(max_depth =1). algorithm = 'SAMME',n_estimators = 200)

clf. fit(x,y)
pred = np.matrix(clf.predict(X))
correct = pred ==Y
Accuracy  = np.sum(correct)/correct.shape[1]
print('Accuracy:' +str(accuracy*100))

#logistic ROC curve and Confustion Matrix
Y = Y.tolist()
pred = pred.T.tolist()

fpr, tpr, thresholds = roc_curve(Y, pred)
plt.plot(fpr)
print.show()

pint(confusion_matrix(Y, pred))
















