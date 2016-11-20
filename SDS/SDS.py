from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
import matplotlib.cm as cm

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)

def plot_feature_set_simple(X, Y, subplot, title):
   
min_x = np.min(X[:, 0])
max_x = np.max(X[:, 0])

min_y = np.min(X[:, 1])
max_y = np.max(X[:, 1])
    
    
#classif = OneVsRestClassifier(SVC(kernel='linear'))
classif = RandomForestClassifier(min_samples_leaf=20)
#n_estimators = 10
#classif = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
classif.fit(cls_data.X,cls_data.Y)

plt.figure(figsize=(8,8))
#plt.subplot(1,1,1)
#plt.title(title) 

colors = cm.rainbow(np.linspace(0, 1, Y.shape[1]))

#block to loop over formulate it first
option1 = '01 - TOP'
option2 = '03 - UNIVERSITARIO'
plt.scatter(cls_data.X.loc[:][option1], cls_data.X.loc[:,option2], s=40, c='grey')
    
zero_class = np.where(cls_data.Y.iloc[:, 1])
one_class = np.where(cls_data.Y.iloc[:, 7])
plt.scatter(cls_data.X.loc[:][option1], cls_data.X.loc[:][option2], s=40, c='gray')
plt.scatter(cls_data.X.loc[zero_class[0]][option1], cls_data.X.iloc[zero_class[0]][option2], s=160, edgecolors='b',
            facecolors='none', linewidths=2, label='Class 1')
plt.scatter(cls_data.X.loc[one_class[0]][option1], cls_data.X.iloc[one_class[0]][option2], s=80, edgecolors='orange',
            facecolors='none', linewidths=2, label='Class 2')

plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                'Boundary\nfor class 1')
plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                'Boundary\nfor class 2')

plt.xticks(())
plt.yticks(())