#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('K:\Fall 2019\MLF\Assignments\HW5\hw5_treasury yield curve data.csv')
df_columns = ['SVENF01','SVENF05','SVENF10','SVENF15','SVENF20','Adj_Close']
cols=df_columns
row=len(df.index)
col=len(df.columns)
print("The number of rows are "+str(row))
print("The number of columns are "+str(col))

#Statistics Summary
stat=df.describe()
print("The Statistics Summary is as follows")
print(stat)

close=["Adj_Close"]
value=['SVENF20']

#Correlation between Target and Attributes
from random import uniform
target=df[close]
attribute=df[value]
plt.scatter(attribute, target)
plt.xlabel("Attribute")
plt.ylabel("Target- Adjusted close")
print("Plot of the Correlation between Target and Attributes")
plt.show()

#Presenting Attribute Correlations Visually
from pandas import DataFrame
corMat = DataFrame(df.corr())
print("The correlation between all the attributes as a heatmap")
plt.pcolor(corMat)
plt.show()

#visualize correlations using heatmap
import seaborn as sns
print("Heat Map")
corr = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
heatmap = sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

#Pearson correlation
from scipy.stats import pearsonr
attribute1=df["SVENF01"]
attribute20=df["SVENF20"]
correlation, pvalue = pearsonr(attribute1, attribute20)
print("The Pearson's correlation between attribute 1 and 20 is %.3f" % correlation)


# In[91]:


#Linear model Regressor and SVM before transforming
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X, y = df.iloc[:, 1:-2].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#Linear Model
from sklearn import linear_model
sgd = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
sgd.fit(X_train, y_train)
print ("Accuracy on training set for a linear model : %.3f" % (sgd.score(X_train, y_train)))
print ("Accuracy on test set for a linear model : %.3f" % (sgd.score(X_test, y_test)))
sgd_scores = cross_val_score(sgd, X_train, y_train, cv=10)
print ("Accuracy on test set using crossvalidation for a linear model: %.3f" % (np.mean(sgd_scores)))
print("The coefficients are : ")
print(sgd.coef_)
y_train_pred = sgd.predict(X_train)
y_test_pred = sgd.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


#SVM with linear kernel
from sklearn import svm
svm_linear = svm.SVR(kernel='linear')
svm_linear.fit(X_train, y_train)
print ("Accuracy on training set for svm (kernel=linear) : %.3f" % (svm_linear.score(X_train, y_train)))
print ("Accuracy on test set for svm (kernel=linear) : %.3f" % (svm_linear.score(X_test, y_test)))
svm_linear_scores = cross_val_score(svm_linear, X_test, y_test, cv=10)
print ("Accuracy on test set using crossvalidation for svm (kernel=linear) : %.3f" % (np.mean(svm_linear_scores)))
print("The coefficients are : ") 
print(svm_linear.coef_)
y_train_pred = svm_linear.predict(X_train)
y_test_pred = svm_linear.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


#SVM with rbf kernel
svm_rbf = svm.SVR(kernel='rbf')
svm_rbf.fit(X_train, y_train)
print ("Accuracy on training set for svm (kernel=rbf) : %.3f" % (svm_rbf.score(X_train, y_train)))
print ("Accuracy on test set for svm (kernel=rbf) : %.3f" % (svm_rbf.score(X_test, y_test)))
svm_rbf_scores = cross_val_score(svm_rbf, X_train, y_train, cv=10)
print ("Accuracy on test set using crossvalidation for svm (kernel=rbf) : %.3f" % (np.mean(svm_rbf_scores)))
y_train_pred = svm_rbf.predict(X_train)
y_test_pred = svm_rbf.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


# In[94]:


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)

#PCA for n_components= None
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print("The explained variance ratio for all components: " + str(pca.explained_variance_ratio_))

#Plot for individual and cumulative explained variance for n_components=0
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,30), var_exp, alpha=0.5, align='center', label='var_exp')
plt.step(range(1,30), cum_var_exp, where='mid', label='cumulative var_exp')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='upper right')
print("Plot for individual and cumulative explained variance for n_components=0")
plt.show()

#PCA for 3 components
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print("The explained variance ratio for n_components=3 is: " +str(pca.explained_variance_ratio_))

#Plot for individual and cumulative explained variance for 3 components
cov_mat = np.cov(X_train_pca.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,4), var_exp, alpha=0.5, align='center', label='var_exp')
plt.step(range(1,4), cum_var_exp, where='mid', label='cumulative var_exp')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='upper right')
print("Plot for individual and cumulative explained variance for n_components=3")
plt.show()


# In[95]:


#Linear Model - After PCA
sgd1 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
sgd1.fit(X_train_pca, y_train)
print ("Accuracy on training set for a linear model- PCA :  %.3f" % (sgd1.score(X_train_pca, y_train)))
print ("Accuracy on test set for a linear model- PCA :  %.3f" % (sgd1.score(X_test_pca, y_test)))
sgd1_scores = cross_val_score(sgd1, X_train_pca, y_train, cv=10)
print ("Accuracy on test set using crossvalidation for a linear model- PCA :  %.3f" % (np.mean(sgd1_scores)))
print("The coefficients are : ")
print(sgd1.coef_)
y_train_pred = sgd1.predict(X_train_pca)
y_test_pred = sgd1.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


#SVM with linear kernel - After PCA
svm_linear1 = svm.SVR(kernel='linear')
svm_linear1.fit(X_train_pca, y_train)
print ("Accuracy on training set for svm (kernel=linear)- PCA :  %.3f" % (svm_linear1.score(X_train_pca, y_train)))
print ("Accuracy on test set for svm (kernel=linear)- PCA :  %.3f" % (svm_linear1.score(X_test_pca, y_test)))
svm_linear1_scores = cross_val_score(svm_linear1, X_test_pca, y_test, cv=10)
print ("Accuracy on test set using crossvalidation for svm (kernel=linear)- PCA :  %.3f" % (np.mean(svm_linear1_scores)))
print("The coefficients are : ") 
print(svm_linear1.coef_)
y_train_pred = svm_linear1.predict(X_train_pca)
y_test_pred = svm_linear1.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


#SVM with rbf kernel - After PCA
svm_rbf1 = svm.SVR(kernel='rbf')
svm_rbf1.fit(X_train_pca, y_train)
print ("Accuracy on training set for svm (kernel=rbf)- PCA :  %.3f" % (svm_rbf1.score(X_train_pca, y_train)))
print ("Accuracy on test set for svm (kernel=rbf)- PCA :  %.3f" % (svm_rbf1.score(X_test_pca, y_test)))
svm_rbf1_scores = cross_val_score(svm_rbf1, X_train_pca, y_train, cv=10)
print ("Accuracy on test set using crossvalidation for svm (kernel=rbf)- PCA :  %.3f" %(np.mean(svm_rbf1_scores)))
y_train_pred = svm_rbf1.predict(X_train_pca)
y_test_pred = svm_rbf1.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

print("My name is Khavya Chandrasekaran")
print("My NetID is: khavyac2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




