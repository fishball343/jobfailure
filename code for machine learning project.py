import requests
from StringIO import StringIO

import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

from sklearn.metrics import f1_score


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn import preprocessing
import sklearn.cross_validation
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

re=pd.read_csv('/Users/Justin/Desktop/alldatas.csv')
re
re1=re.drop('$\{optionToStr(jobFeatures.user)\}', 1)

re2=re1.drop('$\{jobId\}', 1)
re3=re2.drop('$\{optionToStr(jobFeatures.jobName)\}',1)
re4=re3.drop('$\{optionToStr(logicalJobName)\}',1)
re5=re4.drop('endstate',1)
re6=re5.drop('res',1)
Y = re6['RESULT']
re7=re6.drop('RESULT', 1)





Y = [int(x>=1) for x in Y]
Y = np.asarray(Y)
print(re7.shape)
X=re7.as_matrix()


NX=preprocessing.normalize(X) 
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(NX, Y, test_size=0.4, random_state=0)




clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train,y_train)
clf.score(X_test, y_test)

clf2=svm.SVC(kernel='rbf', C=1.0)
clf2.fit(X_train,y_train)
clf2.score(X_test, y_test)


model = LogisticRegression()
model = model.fit(X, Y)
model.score(X, Y)

clf3= GaussianNB()
clf3.fit(X_train, y_train)
clf3.score(X_test, y_test)

clf4= MultinomialNB()
clf4.fit(X_train, y_train)
clf4.score(X_test, y_test)

from sklearn.ensemble import RandomForestClassifier
clf5 = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0)
clf5.fit( X_train, y_train)
clf5.score(X_test, y_test)




#Confusion Matrices

from sklearn.metrics import confusion_matrix
y_true=y_test
y_pred=clf.fit(X_train, y_train)
confusion_matrix(y_true, y_pred)


#LINEAR SUPPORT VECTOR REGRESSION
w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
plt.legend()
plt.show()


#FINDING THE MOST SIGNIFICANT FACTORS

pd.DataFrame(zip(X, np.transpose(model.coef_)))

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(NX, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

#
print("Feature ranking:")
list=[ 'schedulingClass', 'majoritypriority','medianCPUReq','MedianMemreq', 'MedianDiskReq', 'CPU_100', 'CPU_050', 'CPU_025', 'mem_100', 'mem_075',
      'mem_0_50', 'mem_0_25', 'mem_0_12', 'mem_0_04', 'platformA', 'platformB', 'platformC', 'medianCPUMax', 'MedianCPUusage', 'medianMemAlloc',
      'medianMemMax', 'medianMemUsage', 'medianCPI', 'medianMAI', 'medianUnmappedPage CacheMemUsage', 'medianPageCacheMem Usage',
      'memReqxmemAlloc', 'memAllocxmemUsage', 'memUsagexmemMax', 'cpuReqxcpuUsage', 'cpuUsagexcpuMax']
for f in range(15):
   print("%d. feature %s  (%f) %d" % (f + 1, list[indices[f]], importances[indices[f]], indices[f]))

print


