# import necessary libraries
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=columns)

# Split the data
pd.array = df.values
X = pd.array[:, 0:4]  # slicing a 2d array with first 4 items in each dimension
Y = pd.array[:, 4]  # slicing a 2d array with 5th item left in each dimension

# Use of stratified 2-fold cross validation to estimate accuracy
# meaning of split dataframe into 10 parts, train on 9 and test on 1 and
# repeat for all combinations of train-test splits
# set the random seed to a fixed number to ensure each algorithm is evaluated on same splits

# Building models
models = []
models.append(('LR', LinearRegression()))  # linear regression model
models.append(('PR2', LinearRegression()))  # Polynomial regression with degree of 2
models.append(('PR3', LinearRegression()))  # Polynomial regression with degree of 3
models.append(('NB', GaussianNB()))  # Naive Baysian
models.append(('KNN', KNeighborsClassifier()))  # KNeighborsClassifier
models.append(('LDA', LinearDiscriminantAnalysis()))  # LinearDiscriminantAnalysis
models.append(('QDA', QuadraticDiscriminantAnalysis()))  # QuadraticDiscriminantAnalysis

# # Start cross-validation
results = []
names = []
scoring = ['accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error']
# Transform species
Y_trans = Y
Y_trans = np.where(Y=='Iris-setosa', 1, Y_trans)
Y_trans = np.where(Y=='Iris-versicolor', 2, Y_trans)
Y_trans = np.where(Y=='Iris-virginica', 3, Y_trans)

for name, model in models:
    kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
    if name == 'LR':
        cv_result = cross_val_score(model, X, Y_trans, cv=2, scoring='neg_mean_squared_error')
    elif name == 'PR2':
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        cv_result = cross_val_score(model, X_poly, Y_trans, cv=2, scoring='neg_mean_squared_error')
    elif name == 'PR3':
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        cv_result = cross_val_score(model, X_poly, Y_trans, cv=2, scoring='neg_mean_squared_error')
    else:
        cv_result = cross_val_score(model, X, Y, cv=2, scoring='accuracy')
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_result.mean(), cv_result.std()))

