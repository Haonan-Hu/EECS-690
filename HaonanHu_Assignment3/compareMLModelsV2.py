# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=columns)

# Split the data
pd.array = df.values
X = pd.array[:, 0:4]  # slicing a 2d array with first 4 items in each dimension
Y = pd.array[:, 4]  # slicing a 2d array with 5th item left in each dimension
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.5, random_state=1)  # test_size is 50% of totol samples

# Used for calculating accuracy
Y_concat = np.concatenate((Y_train, Y_validation), axis=0)
# Transform species
Y_trans_1 = Y_train
Y_trans_1 = np.where(Y_trans_1 == 'Iris-setosa', 1, Y_trans_1)
Y_trans_1 = np.where(Y_trans_1 == 'Iris-versicolor', 2, Y_trans_1)
Y_trans_1 = np.where(Y_trans_1 == 'Iris-virginica', 3, Y_trans_1)

Y_trans_2 = Y_validation
Y_trans_2 = np.where(Y_trans_2 == 'Iris-setosa', 1, Y_trans_2)
Y_trans_2 = np.where(Y_trans_2 == 'Iris-versicolor', 2, Y_trans_2)
Y_trans_2 = np.where(Y_trans_2 == 'Iris-virginica', 3, Y_trans_2)

# Fit and prediction
# Linear Regression
lr1 = LinearRegression().fit(X_train, Y_trans_1)
prediction_lr1 = lr1.predict(X_validation)  # predict X_validation -> Y_validation
lr2 = LinearRegression().fit(X_validation, Y_trans_2)
prediction_lr2 = lr2.predict(X_train)  # predict X_train -> Y_train
# concatenate to get predicted Y_train + Y_validation
prediction_lr = np.concatenate((prediction_lr2, prediction_lr1), axis=0)
# some necessary data cleansing 
prediction_lr = np.where(prediction_lr < 0.5, 1, prediction_lr)
prediction_lr = np.where(prediction_lr > 3.5, 3, prediction_lr)
prediction_lr = np.rint(prediction_lr).astype(int)
# Transform prediction back to sting
prediction_lr = np.where(prediction_lr == 1, 'Iris-setosa', prediction_lr)
prediction_lr = np.where(prediction_lr == '2', 'Iris-versicolor', prediction_lr)
prediction_lr = np.where(prediction_lr == '3', 'Iris-virginica', prediction_lr)
# calculate accuracy score
accuracy_lr = accuracy_score(Y_concat, prediction_lr)
print(f"The accuracy score for Linear Regression is {accuracy_lr}")
print(confusion_matrix(Y_concat, prediction_lr))
print(classification_report(Y_concat, prediction_lr))

# Polynomial Regression with degree of 2
pr_d2_1 = PolynomialFeatures(degree=2)
X_poly2_1 = pr_d2_1.fit_transform(X_train)
# predict X_validation -> Y_validation
prediction_pr2_1 = LinearRegression().fit(X_poly2_1, Y_trans_1).predict(pr_d2_1.fit_transform(X_validation))
pr_d2_2 = PolynomialFeatures(degree=2)
X_poly2_2 = pr_d2_2.fit_transform(X_validation)
# predict X_train -> Y_train
prediction_pr2_2 = LinearRegression().fit(X_poly2_2, Y_trans_2).predict(pr_d2_2.fit_transform(X_train))
# concatenate to get predicted Y_train + Y_validation
prediction_pr2 = np.concatenate((prediction_pr2_2, prediction_pr2_1), axis=0)
# some necessary data cleansing 
prediction_pr2 = np.where(prediction_pr2 < 0.5, 1, prediction_pr2)
prediction_pr2 = np.where(prediction_pr2 > 3.5, 3, prediction_pr2)
prediction_pr2 = np.rint(prediction_pr2).astype(int)
# Transform prediction back to sting
prediction_pr2 = np.where(prediction_pr2 == 1, 'Iris-setosa', prediction_pr2)
prediction_pr2 = np.where(prediction_pr2 == '2', 'Iris-versicolor', prediction_pr2)
prediction_pr2 = np.where(prediction_pr2 == '3', 'Iris-virginica', prediction_pr2)
# calculate accuracy score
accuracy_pr2 = accuracy_score(Y_concat, prediction_pr2) 
print(f"The accuracy score for Polynomial Regression of degree 2 is {accuracy_pr2}")
print(confusion_matrix(Y_concat, prediction_pr2))
print(classification_report(Y_concat, prediction_pr2))

# Polynomial Regression with degree of 3
pr_d3_1 = PolynomialFeatures(degree=3)
X_poly3_1 = pr_d3_1.fit_transform(X_train)
# predict X_validation -> Y_validation
prediction_pr3_1 = LinearRegression().fit(X_poly3_1, Y_trans_1).predict(pr_d3_1.fit_transform(X_validation))
pr_d3_2 = PolynomialFeatures(degree=3)
X_poly3_2 = pr_d3_2.fit_transform(X_validation)
# predict X_train -> Y_train
prediction_pr3_2 = LinearRegression().fit(X_poly3_2, Y_trans_2).predict(pr_d3_2.fit_transform(X_train))
# concatenate to get predicted Y_train + Y_validation
prediction_pr3 = np.concatenate((prediction_pr3_2, prediction_pr3_1), axis=0)
# some necessary data cleansing 
prediction_pr3 = np.where(prediction_pr3 < 0.5, 1, prediction_pr3)
prediction_pr3 = np.where(prediction_pr3 > 3.5, 3, prediction_pr3)
prediction_pr3 = np.rint(prediction_pr3).astype(int)
# Transform prediction back to sting
prediction_pr3 = np.where(prediction_pr3 == 1, 'Iris-setosa', prediction_pr3)
prediction_pr3 = np.where(prediction_pr3 == '2', 'Iris-versicolor', prediction_pr3)
prediction_pr3 = np.where(prediction_pr3 == '3', 'Iris-virginica', prediction_pr3)
# calculate accuracy score
accuracy_pr3 = accuracy_score(Y_concat, prediction_pr3) 
print(f"The accuracy score for Polynomial Regression of degree 3 is {accuracy_pr3}")
print(confusion_matrix(Y_concat, prediction_pr3))
print(classification_report(Y_concat, prediction_pr3))

# # Naive Bayes
nb_1 = GaussianNB()  
nb_1.fit(X_train, Y_train)
prediction_nb_1 = nb_1.predict(X_validation)  # predict X_validation -> Y_validation
nb_2 = GaussianNB()  
nb_2.fit(X_validation, Y_validation) 
prediction_nb_2 = nb_2.predict(X_train)  # predict X_train -> Y_train
# concatenate to get predicted Y_train + Y_validation
prediction_nb = np.concatenate((prediction_nb_2, prediction_nb_1), axis=0)
accuracy_nb = accuracy_score(Y_concat, prediction_nb)
print(f"The accuracy score for Naive Bayes is {accuracy_nb}")
print(confusion_matrix(Y_concat, prediction_nb))
print(classification_report(Y_concat, prediction_nb))

# # K neighbors
knn_1 = KNeighborsClassifier()  
knn_1.fit(X_train, Y_train)
prediction_knn_1 = knn_1.predict(X_validation)  # predict X_validation -> Y_validation
knn_2 = KNeighborsClassifier()  
knn_2.fit(X_validation, Y_validation)
prediction_knn_2 = knn_2.predict(X_train)  # predict X_train -> Y_train
# concatenate to get predicted Y_train + Y_validation
prediction_knn = np.concatenate((prediction_knn_2, prediction_knn_1), axis=0)
accuracy_knn = accuracy_score(Y_concat, prediction_knn)
print(f"The accuracy score for K Neighbors is {accuracy_knn}")
print(confusion_matrix(Y_concat, prediction_knn))
print(classification_report(Y_concat, prediction_knn))

# # Linear Discriminant Anaylsis
lda_1 = LinearDiscriminantAnalysis()  
lda_1.fit(X_train, Y_train)
prediction_lda_1 = lda_1.predict(X_validation)  # predict X_validation -> Y_validation
lda_2 = LinearDiscriminantAnalysis()  
lda_2.fit(X_validation, Y_validation)
prediction_lda_2 = lda_2.predict(X_train)  # predict X_train -> Y_train
# concatenate to get predicted Y_train + Y_validation
prediction_lda = np.concatenate((prediction_lda_2, prediction_lda_1), axis=0)
accuracy_lda = accuracy_score(Y_concat, prediction_lda)
print(f"The accuracy score for Linear Discriminant Anaylsis is {accuracy_lda}")
print(confusion_matrix(Y_concat, prediction_lda))
print(classification_report(Y_concat, prediction_lda))

# Quadratic Discriminant Anaylsis
qda_1 = QuadraticDiscriminantAnalysis()  
qda_1.fit(X_train, Y_train)
prediction_qda_1 = qda_1.predict(X_validation)  # predict X_validation -> Y_validation
qda_2 = QuadraticDiscriminantAnalysis()  
qda_2.fit(X_validation, Y_validation)
prediction_qda_2 = qda_2.predict(X_train)  # predict X_train -> Y_train
# concatenate to get predicted Y_train + Y_validation
prediction_qda = np.concatenate((prediction_qda_2, prediction_qda_1), axis=0)
accuracy_qda = accuracy_score(Y_concat, prediction_qda)
print(f"The accuracy score for Quadratic Discriminant Anaylsis is {accuracy_qda}")
print(confusion_matrix(Y_concat, prediction_qda))
print(classification_report(Y_concat, prediction_qda))

# Support Vector Machine
# Fitting data into model
linearSVC_1 = svm.LinearSVC(max_iter=10000)
linearSVC_1.fit(X_train, Y_train)
prediction_linearSVC_1 = linearSVC_1.predict(X_validation)  # predict X_validation -> Y_validation
linearSVC_2 = svm.LinearSVC(max_iter=10000)
linearSVC_2.fit(X_validation, Y_validation)
prediction_linearSVC_2 = linearSVC_2.predict(X_train)  # predict X_train -> Y_train
# concatenate to get predicted Y_train + Y_validation
prediction_linearSVC = np.concatenate((prediction_linearSVC_2, prediction_linearSVC_1), axis=0)
accuracy_linearSVC = accuracy_score(Y_concat, prediction_linearSVC)
print(f"The accuracy score for Linear SVC is {accuracy_linearSVC}")
print(confusion_matrix(Y_concat, prediction_linearSVC))
print(classification_report(Y_concat, prediction_linearSVC))

# Decision Tree classifier
decisionTree_1 = DecisionTreeClassifier()
decisionTree_1.fit(X_train, Y_train)
prediction_decisionTree_1 = decisionTree_1.predict(X_validation)
decisionTree_2 = DecisionTreeClassifier()
decisionTree_2.fit(X_validation, Y_validation)
prediction_decisionTree_2 = decisionTree_2.predict(X_train)
# concatenate to get predicted Y_train + Y_validation
prediction_decisionTree = np.concatenate((prediction_decisionTree_2, prediction_decisionTree_1), axis=0)
accuracy_decisionTree = accuracy_score(Y_concat, prediction_decisionTree)
print(f"The accuracy score for Decision Tree is {accuracy_decisionTree}")
print(confusion_matrix(Y_concat, prediction_decisionTree))
print(classification_report(Y_concat, prediction_decisionTree))

# Random Forest Classifier
randomForest_1 = RandomForestClassifier()
randomForest_1.fit(X_train, Y_train)
prediction_randomForest_1 = randomForest_1.predict(X_validation)
randomForest_2 = RandomForestClassifier()
randomForest_2.fit(X_validation, Y_validation)
prediction_randomForest_2 = randomForest_2.predict(X_train)
# concatenate to get predicted Y_train + Y_validation
prediction_randomForest = np.concatenate((prediction_randomForest_2, prediction_randomForest_1), axis=0)
accuracy_randomForest = accuracy_score(Y_concat, prediction_randomForest)
print(f"The accuracy score for Random Forest is {accuracy_randomForest}")
print(confusion_matrix(Y_concat, prediction_randomForest))
print(classification_report(Y_concat, prediction_randomForest))

# Extra Trees Classifier
extraTrees_1 = ExtraTreesClassifier()
extraTrees_1.fit(X_train, Y_train)
prediction_extraTrees_1 = extraTrees_1.predict(X_validation)
extraTrees_2 = ExtraTreesClassifier()
extraTrees_2.fit(X_validation, Y_validation)
prediction_extraTrees_2 = extraTrees_2.predict(X_train)
# concatenate to get predicted Y_train + Y_validation
prediction_extraTrees = np.concatenate((prediction_extraTrees_2, prediction_extraTrees_1), axis=0)
accuracy_extraTrees = accuracy_score(Y_concat, prediction_extraTrees)
print(f"The accuracy score for Extra Trees Classifier is {accuracy_extraTrees}")
print(confusion_matrix(Y_concat, prediction_extraTrees))
print(classification_report(Y_concat, prediction_extraTrees))

# Neural Network MLP Classifier
NN_1 = MLPClassifier(max_iter=1000)
NN_1.fit(X_train, Y_train)
prediction_NN_1 = NN_1.predict(X_validation)
NN_2 = MLPClassifier(max_iter=1000)
NN_2.fit(X_validation, Y_validation)
prediction_NN_2 = NN_2.predict(X_train)
# concatenate to get predicted Y_train + Y_validation
prediction_NN = np.concatenate((prediction_NN_2, prediction_NN_1), axis=0)
accuracy_NN = accuracy_score(Y_concat, prediction_NN)
print(f"The accuracy score for Neural Network MLP Classifier is {accuracy_NN}")
print(confusion_matrix(Y_concat, prediction_NN))
print(classification_report(Y_concat, prediction_NN))
