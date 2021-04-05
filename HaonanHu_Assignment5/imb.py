# import necessary modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks

# Making datafram from csv
iris = pd.read_csv('imbalanced_iris.csv')
# Split the data
pd.array = iris.values
X = pd.array[:, 0:4]  # slicing a 2d array with first 4 items in each dimension
Y = pd.array[:, 4]  # slicing a 2d array with 5th item left in each dimension
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
Y_concat = np.concatenate((Y_train, Y_validation), axis=0)

# Part 1
print("\n\n--------------------------------PART 1----------------------------------------\n\n")
NN_pt1 = MLPClassifier(max_iter=10000)
NN_pt1.fit(X_train, Y_train)
prediction_NNpt1_1 = NN_pt1.predict(X_validation)

NN_pt2 = MLPClassifier(max_iter=10000)
NN_pt2.fit(X_validation, Y_validation)
prediction_NNpt1_2 = NN_pt2.predict(X_train)

# concatenate to get predicted Y_train + Y_validation
prediction_NNpt1 = np.concatenate((prediction_NNpt1_2, prediction_NNpt1_1), axis=0)
accuracy_NNpt1 = accuracy_score(Y_concat, prediction_NNpt1)
matrix = confusion_matrix(Y_concat, prediction_NNpt1)
report = classification_report(Y_concat, prediction_NNpt1, output_dict=True)
print(f"The accuracy score for Neural Network MLP Classifier is {accuracy_NNpt1}")
print(f"The confusion matrix is:\n{matrix}")
# print(classification_report(Y_concat, prediction_NNpt1))


# calculating TP, TN, FP, FN
def specificity_calculate(matrix):
    FP = matrix.sum(axis=0) - np.diag(matrix)  
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)
    specificity = []
    for a in range(matrix.shape[0]):
        tmp = TN[a] / (TN[a] + FP[a])
        specificity.append(tmp)
    return specificity  # return specifity containing each class


# transform matrix
matrix = matrix.T
specificity = specificity_calculate(matrix) 
# calculating precision, recall and specificity for each class
# {class: [precision, recall, specificity]}
performance_metric = {'Iris-setosa': [], 'Iris-versicolor': [], 'Iris-virginica': []}
# retrive percision, recall from classification report and add specification
for key, value in performance_metric.items():
    for k, v in report.items():
        if key == k:
            for a in list(v.values())[:2]:
                value.append(a)
    if key == 'Iris-setosa':
        value.append(specificity[0])
    elif key == 'Iris-versicolor':
        value.append(specificity[1])
    elif key == 'Iris-virginica':
        value.append(specificity[2])

class_balaned_accuracy = 0.0
for k, v in performance_metric.items():
    class_balaned_accuracy += np.minimum(v[0], v[1])
class_balaned_accuracy = class_balaned_accuracy / (matrix.shape[0])
print(f"The class balanced accuracy is {class_balaned_accuracy}")

balanced_accuracy = 0.0
for k, v in performance_metric.items():
    balanced_accuracy += ((v[1] + v[2]) / 2)
balanced_accuracy = balanced_accuracy / (matrix.shape[0])
print(f"The balanced accuracy is {balanced_accuracy}")

balanced_accuracy_score = balanced_accuracy_score(Y_concat, prediction_NNpt1)
print(f"The balanced accuracy score by skikit-learn is {balanced_accuracy_score}")


# Part 2
print("\n\n--------------------------------PART 2----------------------------------------\n\n")
# Resampling using random oversampling
X_resampled_ros, Y_resampled_ros = RandomOverSampler(random_state=0).fit_resample(X, Y)
X_train_ros, X_validation_ros, Y_train_ros, Y_validation_ros = train_test_split(X_resampled_ros, Y_resampled_ros, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
Y_concat_ros = np.concatenate((Y_train_ros, Y_validation_ros), axis=0)
# Fit model using NN
NN_ros1 = MLPClassifier(max_iter=10000)
NN_ros1.fit(X_train_ros, Y_train_ros)
prediction_NN_ros1 = NN_ros1.predict(X_validation_ros)
NN_ros2 = MLPClassifier(max_iter=10000)
NN_ros2.fit(X_validation_ros, Y_validation_ros)
prediction_NN_ros2 = NN_ros2.predict(X_train_ros)
# concatenate to get predicted Y_train + Y_validation
prediction_NN_ros = np.concatenate((prediction_NN_ros2, prediction_NN_ros1), axis=0)
accuracy_NN_ros = accuracy_score(Y_concat_ros, prediction_NN_ros)
print(f"The accuracy score for random oversampling is {accuracy_NN_ros}")
print(confusion_matrix(Y_concat_ros, prediction_NN_ros))
print('\n')

# Resampling using SMOTE
X_resampled_smote, Y_resampled_smote = SMOTE().fit_resample(X, Y)
X_train_smote, X_validation_smote, Y_train_smote, Y_validation_smote = train_test_split(X_resampled_smote, Y_resampled_smote, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
Y_concat_smote = np.concatenate((Y_train_smote, Y_validation_smote), axis=0)
# Fit model using NN
NN_smote1 = MLPClassifier(max_iter=10000)
NN_smote1.fit(X_train_smote, Y_train_smote)
prediction_NN_smote1 = NN_smote1.predict(X_validation_smote)
NN_smote2 = MLPClassifier(max_iter=10000)
NN_smote2.fit(X_validation_smote, Y_validation_smote)
prediction_NN_smote2 = NN_smote2.predict(X_train_smote)
# concatenate to get predicted Y_train + Y_validation
prediction_NN_smote = np.concatenate((prediction_NN_smote2, prediction_NN_smote1), axis=0)
accuracy_NN_smote = accuracy_score(Y_concat_smote, prediction_NN_smote)
print(f"The accuracy score for SMOTE oversampling is {accuracy_NN_smote}")
print(confusion_matrix(Y_concat_smote, prediction_NN_smote))
print('\n')

# Resampling using ADASYN
X_resampled_adasyn, Y_resampled_adasyn = ADASYN(sampling_strategy='minority').fit_resample(X, Y)
X_train_adasyn, X_validation_adasyn, Y_train_adasyn, Y_validation_adasyn = train_test_split(X_resampled_adasyn, Y_resampled_adasyn, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
Y_concat_adasyn = np.concatenate((Y_train_adasyn, Y_validation_adasyn), axis=0)
# Fit model using NN
NN_adasyn1 = MLPClassifier(max_iter=10000)
NN_adasyn1.fit(X_train_adasyn, Y_train_adasyn)
prediction_NN_adasyn1 = NN_adasyn1.predict(X_validation_adasyn)
NN_adasyn2 = MLPClassifier(max_iter=10000)
NN_adasyn2.fit(X_validation_adasyn, Y_validation_adasyn)
prediction_NN_adasyn2 = NN_adasyn2.predict(X_train_adasyn)
# concatenate to get predicted Y_train + Y_validation
prediction_NN_adasyn = np.concatenate((prediction_NN_adasyn2, prediction_NN_adasyn1), axis=0)
accuracy_NN_adasyn = accuracy_score(Y_concat_adasyn, prediction_NN_adasyn)
print(f"The accuracy score for ADASYN oversampling is {accuracy_NN_adasyn}")
print(confusion_matrix(Y_concat_adasyn, prediction_NN_adasyn))


# Part 3
print("\n\n--------------------------------PART 3----------------------------------------\n\n")
# Undersampling using Random undersamper
X_resampled_rus, Y_resampled_rus = RandomUnderSampler(random_state=0, replacement=True).fit_resample(X, Y)
X_train_rus, X_validation_rus, Y_train_rus, Y_validation_rus = train_test_split(X_resampled_rus, Y_resampled_rus, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
Y_concat_rus = np.concatenate((Y_train_rus, Y_validation_rus), axis=0)
# Fit model using NN
NN_rus1 = MLPClassifier(max_iter=10000)
NN_rus1.fit(X_train_rus, Y_train_rus)
prediction_NN_rus1 = NN_rus1.predict(X_validation_rus)
NN_rus2 = MLPClassifier(max_iter=10000)
NN_rus2.fit(X_validation_rus, Y_validation_rus)
prediction_NN_rus2 = NN_rus2.predict(X_train_rus)
# concatenate to get predicted Y_train + Y_validation
prediction_NN_rus = np.concatenate((prediction_NN_rus2, prediction_NN_rus1), axis=0)
accuracy_NN_rus = accuracy_score(Y_concat_rus, prediction_NN_rus)
print(f"The accuracy score for Random undersampling is {accuracy_NN_rus}")
print(confusion_matrix(Y_concat_rus, prediction_NN_rus))
print('\n')

# Undersampling using Cluster undersampling
X_resampled_cluster, Y_resampled_cluster = ClusterCentroids(random_state=0).fit_resample(X, Y)
X_train_cluster, X_validation_cluster, Y_train_cluster, Y_validation_cluster = train_test_split(X_resampled_cluster, Y_resampled_cluster, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
Y_concat_cluster = np.concatenate((Y_train_cluster, Y_validation_cluster), axis=0)
# Fit model using NN
NN_cluster1 = MLPClassifier(max_iter=10000)
NN_cluster1.fit(X_train_cluster, Y_train_cluster)
prediction_NN_cluster1 = NN_cluster1.predict(X_validation_cluster)
NN_cluster2 = MLPClassifier(max_iter=10000)
NN_cluster2.fit(X_validation_cluster, Y_validation_cluster)
prediction_NN_cluster2 = NN_cluster2.predict(X_train_cluster)
# concatenate to get predicted Y_train + Y_validation
prediction_NN_cluster = np.concatenate((prediction_NN_cluster2, prediction_NN_cluster1), axis=0)
accuracy_NN_cluster = accuracy_score(Y_concat_cluster, prediction_NN_cluster)
print(f"The accuracy score for Cluster undersampling is {accuracy_NN_cluster}")
print(confusion_matrix(Y_concat_cluster, prediction_NN_cluster))
print('\n')

# Undersampling using Tomek Link
X_resampled_tl, Y_resampled_tl = TomekLinks().fit_resample(X, Y)
X_train_tl, X_validation_tl, Y_train_tl, Y_validation_tl = train_test_split(X_resampled_tl, Y_resampled_tl, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
Y_concat_tl = np.concatenate((Y_train_tl, Y_validation_tl), axis=0)
# Fit model using NN
NN_tl1 = MLPClassifier(max_iter=10000)
NN_tl1.fit(X_train_tl, Y_train_tl)
prediction_NN_tl1 = NN_tl1.predict(X_validation_tl)
NN_tl2 = MLPClassifier(max_iter=10000)
NN_tl2.fit(X_validation_tl, Y_validation_tl)
prediction_NN_tl2 = NN_tl2.predict(X_train_tl)
# concatenate to get predicted Y_train + Y_validation
prediction_NN_tl = np.concatenate((prediction_NN_tl2, prediction_NN_tl1), axis=0)
accuracy_NN_tl = accuracy_score(Y_concat_tl, prediction_NN_tl)
print(f"The accuracy score for TomekLinks undersampling is {accuracy_NN_tl}")
print(confusion_matrix(Y_concat_tl, prediction_NN_tl))
print('\n')
