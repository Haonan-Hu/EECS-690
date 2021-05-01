# import necessary lib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# import dataset 
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = pd.read_csv(url, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])

# Split the data
pd.array = df.values
X = pd.array[:, 0:4]  # slicing a 2d array with first 4 items in each dimension
Y = pd.array[:, 4]  # slicing a 2d array with 5th item left in each dimension

Y_trans = Y
Y_trans = np.where(Y_trans == 'Iris-setosa', 0, Y_trans)
Y_trans = np.where(Y_trans == 'Iris-versicolor', 1, Y_trans)
Y_trans = np.where(Y_trans == 'Iris-virginica', 2, Y_trans)


# Part 1
print("\n\n-------------------K mean clustering-------------------\n\n")
k = [a for a in range(1, 21)]
reconstruction_error = []

# insert construction_error for k = 1 to 20
for i in k:
    reconstruction_error.append(KMeans(n_clusters=i).fit(X).inertia_)

fig, (ax1, ax2, ax3) = plt.subplots(3)
plt.subplots_adjust(hspace=1.0)
fig.suptitle('Reconstruction Error and AIC vs k')
ax1.plot(k, reconstruction_error)
ax1.set(title='ReconstructionError vs k',
        xlabel='k',
        ylabel='Reconstruction Error',
        xticks=k)


print("By observing the graph, the elbow is when k = 3\n\n")
elbow_k = 3
kmean = KMeans(n_clusters=elbow_k).fit(X)
prediction_k1 = kmean.predict(X)
# transform prediction to species
prediction_k1 = np.rint(prediction_k1).astype(int)
prediction_k1 = np.where(prediction_k1 == 0, 'Iris-setosa', prediction_k1)
prediction_k1 = np.where(prediction_k1 == '1', 'Iris-versicolor', prediction_k1)
prediction_k1 = np.where(prediction_k1 == '2', 'Iris-virginica', prediction_k1)

k_labels = prediction_k1
k_labels_matched = np.empty(k_labels.shape)
# sorting
for k in np.unique(k_labels):
    match_nums = [np.sum((k_labels == k) * (Y_trans == t)) for t in np.unique(Y_trans)]
    k_labels_matched[k_labels == k] = np.unique(Y_trans)[np.argmax(match_nums)]
# transform 
k_labels_matched = np.rint(k_labels_matched).astype(int)
k_labels_matched = np.where(k_labels_matched == 0, 'Iris-setosa', k_labels_matched)
k_labels_matched = np.where(k_labels_matched == '1', 'Iris-versicolor', k_labels_matched)
k_labels_matched = np.where(k_labels_matched == '2', 'Iris-virginica', k_labels_matched)
# prediction_k = model.predict(k_labels_matched.reshape(-1, 1))
matrix_k = confusion_matrix(Y, k_labels_matched)
accuracy_k = accuracy_score(Y, k_labels_matched)
print(f"The accuracy score for K-mean clustering using elbow_k is {accuracy_k}")
print(confusion_matrix(Y, k_labels_matched))
print(f"The accuracy score for K-mean clustering using k = 3 is {accuracy_k}")
print(confusion_matrix(Y, k_labels_matched))

# # Part 2
print("\n\n-------------------Gaussian Mixture Models(GMM)-------------------\n\n")
# AIC
k = [a for a in range(1, 21)]
aic = []

for i in k:
    aic.append(GaussianMixture(n_components=i, covariance_type='diag').fit(X).aic(X))

ax2.plot(k, aic)
ax2.set(title='AIC vs k',
        xlabel='k',
        ylabel='AIC',
        xticks=k)
print("By observing the graph, the elbow of AIC graph is when k = 3\n\n")
aic_elbow_k = 3
# BIC
bic = []
for i in k:
    bic.append(GaussianMixture(n_components=i, covariance_type='diag').fit(X).bic(X))
ax3.plot(k, bic)
ax3.set(title='BIC vs k',
        xlabel='k',
        ylabel='BIC',
        xticks=k)
plt.show()
plt.close()
print("By observing the graph, the elbow of BIC graph is when k = 3\n\n")
bic_elbow_k = 3

# starte predicting
gmm = GaussianMixture(n_components=aic_elbow_k, covariance_type='diag').fit(X)
prediction_gmm_aic = gmm.predict(X)
# transform prediction to species
prediction_gmm_aic = np.rint(prediction_gmm_aic).astype(int)
prediction_gmm_aic = np.where(prediction_gmm_aic == 0, 'Iris-setosa', prediction_gmm_aic)
prediction_gmm_aic = np.where(prediction_gmm_aic == '1', 'Iris-versicolor', prediction_gmm_aic)
prediction_gmm_aic = np.where(prediction_gmm_aic == '2', 'Iris-virginica', prediction_gmm_aic)

k_labels = prediction_gmm_aic
k_labels_matched = np.empty(k_labels.shape)
# sorting
for k in np.unique(k_labels):
    match_nums = [np.sum((k_labels == k) * (Y_trans == t)) for t in np.unique(Y_trans)]
    k_labels_matched[k_labels == k] = np.unique(Y_trans)[np.argmax(match_nums)]
# transform 
k_labels_matched = np.rint(k_labels_matched).astype(int)
k_labels_matched = np.where(k_labels_matched == 0, 'Iris-setosa', k_labels_matched)
k_labels_matched = np.where(k_labels_matched == '1', 'Iris-versicolor', k_labels_matched)
k_labels_matched = np.where(k_labels_matched == '2', 'Iris-virginica', k_labels_matched)
# prediction_k = model.predict(k_labels_matched.reshape(-1, 1))
matrix_gmm_aic = confusion_matrix(Y, k_labels_matched)
accuracy_gmm_aic = accuracy_score(Y, k_labels_matched)
print(f"The accuracy score for GMM using aic_elbow_k is {accuracy_gmm_aic}")
print(confusion_matrix(Y, k_labels_matched))

gmm = GaussianMixture(n_components=bic_elbow_k, covariance_type='diag').fit(X)
prediction_gmm_bic = gmm.predict(X)
# transform prediction to species
prediction_gmm_bic = np.rint(prediction_gmm_bic).astype(int)
prediction_gmm_bic = np.where(prediction_gmm_bic == 0, 'Iris-setosa', prediction_gmm_bic)
prediction_gmm_bic = np.where(prediction_gmm_bic == '1', 'Iris-versicolor', prediction_gmm_bic)
prediction_gmm_bic = np.where(prediction_gmm_bic == '2', 'Iris-virginica', prediction_gmm_bic)

k_labels = prediction_gmm_bic
k_labels_matched = np.empty(k_labels.shape)
# sorting
for k in np.unique(k_labels):
    match_nums = [np.sum((k_labels == k) * (Y_trans == t)) for t in np.unique(Y_trans)]
    k_labels_matched[k_labels == k] = np.unique(Y_trans)[np.argmax(match_nums)]
# transform 
k_labels_matched = np.rint(k_labels_matched).astype(int)
k_labels_matched = np.where(k_labels_matched == 0, 'Iris-setosa', k_labels_matched)
k_labels_matched = np.where(k_labels_matched == '1', 'Iris-versicolor', k_labels_matched)
k_labels_matched = np.where(k_labels_matched == '2', 'Iris-virginica', k_labels_matched)
# prediction_k = model.predict(k_labels_matched.reshape(-1, 1))
matrix_gmm_bic = confusion_matrix(Y, k_labels_matched)
accuracy_gmm_bic = accuracy_score(Y, k_labels_matched)
print(f"The accuracy score for GMM using bic_elbow_k is {accuracy_gmm_bic}")
print(confusion_matrix(Y, k_labels_matched))

print(f"The accuracy score for GMM using k = 3 is {accuracy_gmm_bic}")
print(confusion_matrix(Y, k_labels_matched))
