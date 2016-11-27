from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from sklearn import neighbors
from sklearn import svm
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import parallel_coordinates

# Would be good to produce a couple of the parallel_coordinate plots or the andrews_curves plots
# to demonstrate which of the eigenvalues to use

pca = PCA(n_components=5)
X_pca = pca.fit_transform(cls_data.scaling(X)) 
X_test_pca = pca.transform(X_test)


for i in range(0, len(X.columns.values)):
    if len(X.iloc[:,i].unique()) > 2:
        print(X.columns.values[i])

import numpy as np
import matplotlib.pyplot as plt


colors = cm.rainbow(np.linspace(0, 1, Y.shape[1]))


for i in range(0, Y.shape[1]):
    ith_class = np.where(Y.iloc[:,i])
    plt.scatter(X_pca[ith_class[0],0], X_pca[ith_class[0],1], s=80, edgecolors=colors[i],
            facecolors='none', linewidths=2, label='Class %s'%str(i))

one_class = np.where(y_binary)
#one_class = np.where(Y.loc[:, 'ind_hip_fin_ult1'])
plt.figure(figsize=(10,10))
plt.ylim(-10,10)
plt.scatter(X_pca[:,0], X_pca[:, 1], s=40, c='gray')
plt.scatter(X_pca[one_class[0],0], X_pca[one_class[0],1], s=160, edgecolors='b',
           facecolors='none', linewidths=2, label='True')

plt.scatter(X_pca[~one_class[0],0], X_pca[~one_class[0],1], s=80, edgecolors='orange',
           facecolors='none', linewidths=2, label='False')

y_binary = [f_(sum(r)) for i, r in Y.iterrows()]
y_binary = [sum(r) for i, r in Y.iterrows()]

def f_(sum_val):
    x = -1
    if sum_val <= 1:
        x = 1
    elif sum_val>1 and sum_val < 7:
        x = 2
    else:
        x = 3
    return x

y = np.array(y_binary)

colours_mesh = cm.rainbow(np.linspace(0,1,13))
h = 0.02
C = 1.0
svc = svm.SVC(kernel='linear', C = C, cache_size=2000).fit(X_pca[:,:2], y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_pca, y)
poly_svc = svm.SVC(kernel='poly', degree=4, C = C).fit(X_pca, y)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
w_min, w_max = X_pca[:, 2].min() - 1, X_pca[:, 2].max() + 1
v_min, v_max = X_pca[:, 3].min() - 1, X_pca[:, 3].max() + 1

xx, yy, ww, vv = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h),
                     np.arange(w_min, w_max, h),
                     np.arange(v_min, v_max, h))

for i, clf in enumerate((svc, rbf_svc, poly_svc)):
    plt.subplot(2,2, i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_pca[:,0],X_pca[:,1],c=y, cmap=plt.cm.coolwarm)
    plt.ylim(yy.min(), yy.max())
    

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_pca[:,0],X_pca[:,1],c=y, cmap=plt.cm.coolwarm)
plt.ylim(yy.min(), yy.max())

andrews_curves(X_pca, y)

X.loc[:,['ncodpers','age','antiguedad','days_recorded_from_ny','tenure','days_churned','earnings']]
    
for i in Y_test.columns.values:
    print(Y_test[i].value_counts())

S = set(X)

T = set(X_train)

S.difference(T)
T.difference(S)

