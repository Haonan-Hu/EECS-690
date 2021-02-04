# import necessary libraries
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=columns)

print(df.shape, '\n')  # print the property of the dataframe
print(df.head(20), '\n')  # print the first 20 enties of the dataframe
print(df.describe(), '\n')  # print the statistical description of the dataframe
print(df.groupby('class').size(), '\n')  # print the number of entries for each class

# Univariate plot
# whisker plot
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histgram
df.hist()
plt.show()

# Multivariate plot
# scatter plot
pd.plotting.scatter_matrix(df)
plt.show()

# Split the data
pd.array = df.values
X = pd.array[:,0:4]  # slicing a 2d array with first 4 items in each dimension
Y = pd.array[:,4]  # slicing a 2d array with 5th item left in each dimension
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)  # test_size is 20% of totol samples

# Use of stratified 10-fold cross validation to estimate accuracy
# meaning of split dataframe into 10 parts, train on 9 and test on 1 and
# repeat for all combinations of train-test splits
# set the random seed to a fixed number to ensure each algorithm is evaluated on same splits

# Building models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate modes
result = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    result.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

print("")

# comparing algorithms with whisker plot
plt.boxplot(result, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# making prediction
predict_model = SVC(gamma='auto')
predict_model.fit(X_train, Y_train)
prediction = predict_model.predict(X_validation)

# Calculate accuracy
print(f"The accuracy score is {accuracy_score(Y_validation, prediction)}")
print(confusion_matrix(Y_validation, prediction))
print(classification_report(Y_validation, prediction))
