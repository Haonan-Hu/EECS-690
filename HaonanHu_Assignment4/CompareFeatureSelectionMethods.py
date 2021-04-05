# import necessary libraries
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from itertools import combinations


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=columns)

# Split the data
pd.array = df.values
X = pd.array[:, 0:4]  # slicing a 2d array with first 4 items in each dimension
Y = pd.array[:, 4]  # slicing a 2d array with 5th item left in each dimension

# Split the data
pd.array = df.values
X = pd.array[:, 0:4]  # slicing a 2d array with first 4 items in each dimension
Y = pd.array[:, 4]  # slicing a 2d array with 5th item left in each dimension
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
# Used for calculating accuracy
Y_concat = np.concatenate((Y_train, Y_validation), axis=0)


# Part1
print("\n\n--------------------------------PART 1----------------------------------------\n\n")
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
print(f"Features used: {columns[0:-1]}")


# Part2
print("\n\n--------------------------------PART 2----------------------------------------\n\n")
c_features_dict = dict({0: 'z1', 1: 'z2', 2: 'z3', 3: 'z4'})
A = X
A_mean = np.mean(A.T, axis=1)  # mean of each column
C = A - A_mean  # Centered matrix
C_cov = np.cov(C.T.astype(float))  # calculate the covariance of Centered matrix
values, vectors = np.linalg.eig(C_cov)  # eigen decomposition
print(f"eigen values:\n{values}\neigen vectors: \n{vectors}")
z = vectors.T.dot(C.T)
temp = np.cumsum(values)
count = 0 
for a in temp:
    pov = a / sum(values)
    count += 1
    if pov > 0.9:
        print(f"PoV: {pov}")
        break

X_transform = z[:count]
Y_transform = vectors[:count]
feature_subset = [c_features_dict.get(key) for key in range(count)]
print(f"Subset of transformed features: {X_transform}\n")

# Making train test split
X_train_pt2, X_validation_pt2, Y_train_pt2, Y_validation_pt2 = train_test_split(X_transform.T, Y, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
# Used for calculating accuracy
Y_concat_pt2 = np.concatenate((Y_train_pt2, Y_validation_pt2), axis=0)

# Decision Tree classifier
pt2_1 = DecisionTreeClassifier()
pt2_1.fit(X_train_pt2, Y_train_pt2)
prediction_pt2_1 = pt2_1.predict(X_validation_pt2)
pt2_2 = DecisionTreeClassifier()
pt2_2.fit(X_validation_pt2, Y_validation_pt2)
prediction_pt2_2 = pt2_2.predict(X_train_pt2)
# print(X_train_pt2.shape, Y_train_pt2.shape)
# concatenate to get predicted Y_train + Y_validation
prediction_pt2 = np.concatenate((prediction_pt2_2, prediction_pt2_1), axis=0)
accuracy_decisionTree = accuracy_score(Y_concat_pt2, prediction_pt2)
print(f"The accuracy score for part 2 is {accuracy_decisionTree}")
print(confusion_matrix(Y_concat_pt2, prediction_pt2))
print(f"Subset of Features used: {feature_subset}")

# Part 3
print("\n\n--------------------------------PART 3----------------------------------------\n\n")
total_features = np.concatenate((X, z.T), axis=1)  # total of 8 features in array
features_dict = dict({0: 'sepal-length', 1: 'sepal-width', 2: 'petal-length', 3: 'petal-width', 4: 'z1', 5: 'z2', 6: 'z3', 7: 'z4'}) 
current = [0, 1, 2, 3, 4, 5, 6, 7]
cumulated_accuracy = []
c = 1
restart = 0
pr_accept = []
status_report = pd.DataFrame(columns=['iteration', 'Set of Features', 'Accuracy', 'Pr[Accept]', 'Random Uniform', 'Status'])
best_set = []
best_accuracy = 0


def final_fit(chosen_set):
    X_train_best, X_validation_best, Y_train_best, Y_validation_best = train_test_split(total_features[:, chosen_set], Y, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
    # Used for calculating accuracy
    Y_concat_best = np.concatenate((Y_train_best, Y_validation_best), axis=0)
    decisionTree_1 = DecisionTreeClassifier()
    decisionTree_1.fit(X_train_best, Y_train_best)
    prediction_decisionTree_1 = decisionTree_1.predict(X_validation_best)
    decisionTree_2 = DecisionTreeClassifier()
    decisionTree_2.fit(X_validation_best, Y_validation_best)
    prediction_decisionTree_2 = decisionTree_2.predict(X_train_best)
    # concatenate to get predicted Y_train + Y_validation
    prediction_decisionTree = np.concatenate((prediction_decisionTree_2, prediction_decisionTree_1), axis=0)
    accuracy_decisionTree = accuracy_score(Y_concat_best, prediction_decisionTree)
    print(f"The accuracy score for best set is {accuracy_decisionTree}")
    print(confusion_matrix(Y_concat_best, prediction_decisionTree))
    print(f"Features used: {get_features(chosen_set)}")


# perturb the current feature subset
def perturb(current_subset):
    # print('current shape', current_subset.shape[1])
    # current_size = current_subset.shape[1]
    random_choice = random.randint(0, 1)  # randomly add/delete
    random_num = random.randint(1, 2)  # number to add or delete
    current_subset = action(current_subset, random_choice, random_num)
    current_subset.sort()
    return current_subset


# add or delete
def action(current_subset, choice, iter):
    # add
    if choice == 1:
        for _ in range(iter):
            random_index = random.randint(0, 7)
            # stop adding if full
            if len(current_subset) == 8:
                if random_index in current_subset:
                    current_subset.remove(random_index)
                    if len(current_subset) == 0:  # if after remove the set is empty, add one feature to make sure at least one feature in the set
                        action(current_subset, 1, 1)   
                # print(f"cant add anymore {random_index} {current}")
            else:
                if random_index not in current_subset:
                    current_subset.append(random_index)
                    # print(f"add {random_index} {current}")
                else:
                    continue
    else:
        for _ in range(iter):
            random_index = random.randint(0, 7)
            # stop deleting if empty
            if len(current_subset) == 0:
                current_subset.append(random_index)
                # print(f"empty, now adding {random_index} {current}")
            else:
                if random_index in current_subset:
                    current_subset.remove(random_index)
                    if len(current_subset) == 0:
                        action(current_subset, random.randint(0, 1), 1)
                    # print(f"delete {random_index} {current}") 
                else:
                    continue
    return current_subset


# fit model and estimate performance with 2-fold cross validation
def fit_model(current_subset, accuracy):
    # Making train test split
    X_train_pt3, X_validation_pt3, Y_train_pt3, Y_validation_pt3 = train_test_split(current_subset, Y, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
    # Used for calculating accuracy
    Y_concat_pt3 = np.concatenate((Y_train_pt3, Y_validation_pt3), axis=0)

    pt3_1 = DecisionTreeClassifier()
    pt3_1.fit(X_train_pt3, Y_train_pt3)
    prediction_pt3_1 = pt3_1.predict(X_validation_pt3)
    pt3_2 = DecisionTreeClassifier()
    pt3_2.fit(X_validation_pt3, Y_validation_pt3)
    prediction_pt3_2 = pt3_2.predict(X_train_pt3)
    # concatenate to get predicted Y_train + Y_validation
    prediction_pt3 = np.concatenate((prediction_pt3_2, prediction_pt3_1), axis=0)
    accuracy_pt3 = accuracy_score(Y_concat_pt3, prediction_pt3)
    accuracy.append(accuracy_pt3)
    # print(f"The accuracy score for iteration {i} is {accuracy_pt3}")


# get subset of features
def get_features(current_subset):
    temp_str = ""
    values = features_dict.values()
    values_list = list(values)
    tmp_list = []
    for a in current_subset:
        tmp_list.append(values_list[a])
    temp_str = ', '.join(tmp_list)
    return temp_str


# print report
def report(current_subset, iteration, status, pr='', random_uniform=''):
    global status_report
    temp_arr = np.array([[iteration, get_features(current_subset), cumulated_accuracy[-1], pr, random_uniform, status]])
    temp_df = pd.DataFrame(data=temp_arr, columns=['iteration', 'Set of Features', 'Accuracy', 'Pr[Accept]', 'Random Uniform', 'Status'])
    status_report = status_report.append(temp_df, ignore_index=True)


# Main loop for SA
for i in range(100):
    current = perturb(current)  # the the current index list
    temp_arr = total_features[:, current]  # pull array based on index
    fit_model(temp_arr, cumulated_accuracy)
    if len(cumulated_accuracy) >= 2:
        # improved with if most recent accuracy is higher, Accept
        if cumulated_accuracy[-1] > cumulated_accuracy[-2]:
            report(current, i, 'Improved')
        else:
            random_uni = np.random.uniform()
            pr_accept = np.exp((-i)*((cumulated_accuracy[-2] - cumulated_accuracy[-1]) / cumulated_accuracy[-2]))
            # Accept if Random Uniform <= Pr[Accept]
            if random_uni > pr_accept:
                report(current, i, 'Rejected', pr=pr_accept, random_uniform=random_uni)
            # Reject if Random Uniform > Pr[Accept]
            else:
                report(current, i, 'Accepted', pr=pr_accept, random_uniform=random_uni)
    else:
        # improved
        report(current, i, 'Improved')

    if cumulated_accuracy[-1] > best_accuracy:
        best_accuracy = cumulated_accuracy[-1]
        best_set.append(list(current))
        if i != 0:
            status_report.iloc[i].update({'Status': 'Restart'})
        restart = 0
    else:
        restart += 1
        if restart == 10:
            current = list(best_set[-1])
            status_report.iloc[i].update({'Status': 'Restart'})
            restart = 0
# print(i, current, '\t\t\t', best_set)
print(tabulate(status_report, headers='keys', tablefmt='psql', showindex=False))
print(f"The best set: [{get_features(best_set[-1])}], ")
final_fit(best_set[-1])

# Part 4
print("\n\n--------------------------------PART 4----------------------------------------\n\n")
initial_population = [[4, 0, 1, 2, 3], [4, 5, 1, 2, 3], [4, 5, 6, 1, 2], [4, 5, 6, 7, 1], [4, 5, 6, 7, 0]]
selected_set = []
crossover_set = []
mutated_set = []
ga_accuracy = []
generation = 1


# get unique list of accuracy
def unique_accuracy(accuracy_list):
    res = []
    for item in accuracy_list: 
        if item not in res: 
            res.append(item)
    return res


# Union two list while w/o replication but maintaining ordering
def union(list1, list2):
    return list(set(list1) | set(list2))


# Intersection of two lists
def intersection(list1, list2):
    return list(set(list1) & set(list2))


# fit model with 2-fold cross validation
def evaulate(individual):
    # Making train test split
    X_train_pt4, X_validation_pt4, Y_train_pt4, Y_validation_pt4 = train_test_split(individual, Y, test_size=0.5, random_state=1)  # test_size is 50% of totol samples
    # Used for calculating accuracy
    Y_concat_pt4 = np.concatenate((Y_train_pt4, Y_validation_pt4), axis=0)

    pt4_1 = DecisionTreeClassifier()
    pt4_1.fit(X_train_pt4, Y_train_pt4)
    prediction_pt4_1 = pt4_1.predict(X_validation_pt4)
    pt4_2 = DecisionTreeClassifier()
    pt4_2.fit(X_validation_pt4, Y_validation_pt4)
    prediction_pt4_2 = pt4_2.predict(X_train_pt4)
    # concatenate to get predicted Y_train + Y_validation
    prediction_pt4 = np.concatenate((prediction_pt4_2, prediction_pt4_1), axis=0)
    accuracy_pt4 = accuracy_score(Y_concat_pt4, prediction_pt4)
    return accuracy_pt4
    # print(f"The accuracy score for iteration {i} is {accuracy_pt4}")


# Combine features
def crossover(individual):
    total_union = []
    total_intersection = []
    result = []
   
    com = combinations(individual, 2)
    for a in com:
        total_intersection.append(intersection(a[0], a[1]))
        total_union.append(union(a[0], a[1]))
    result = total_union + total_intersection
    return result


# Improve overall feature pool with mutation
def mutation(individual):
    individual.sort()
    result = []
    
    choice = random.randint(0, 2)  # 0 = add, 1 = delete, 2 = replacing
    s = [random.randint(0, 7)]  # a random feature from feature set
    if len(individual) != 0:
        f = [random.choice(individual)]  # a random feature from individual
    else:
        choice = 0
    # operation for different situation
    if choice == 0:
        if s not in individual:
            result = union(individual, s)
        else:
            result = [item for item in individual if item not in f]  # remove feature f
            if len(result) == 0:
                result = union(s, result)
    elif choice == 1:
        result = [item for item in individual if item not in f]  # remove feature f
        if len(result) == 0:
            result = union(s, result)
    elif choice == 2:
        result = [s if i == f else i for i in individual]
    return result


# crossover(temp_population)
# Main loop
while generation <= 50:
    if generation == 1:
        # making 50 sets
        selected_set = initial_population
        crossover_set = crossover(selected_set)
        # mutate the selected_set
        for a in (selected_set + crossover_set):
            mutated_set.append(mutation(a))
        temp_total = selected_set + crossover_set + mutated_set
        # evaulate
        for i in temp_total:
            temp_data = total_features[:, i]
            ga_accuracy.append((i, evaulate(temp_data)))
        ga_accuracy.sort(key=lambda x: x[1])  # sort accuracy
        # check if 100% accuracy is in first generation
        for i in ga_accuracy:
            if i[1] == 1.0:
                print(f"solution found, set:{get_features(i[0])} has accuracy: {i[1]}")  # probably will never hit
    else:
        # let top 5 accuracy set as selected set from generation 2
        selected_set = [a[0] for a in ga_accuracy[-5:0]]
        # repeat until 50 generatoin is finished
        crossover_set = crossover(selected_set)
        # mutate the selected_set
        temp_mutated = []
        for a in selected_set + crossover_set:
            temp_mutated.append(mutation(a))
        mutated_set = temp_mutated
        temp_total = selected_set + crossover_set + mutated_set
        # evaulate
        # print(len(selected_set), len(crossover_set), len(mutated_selected_set), len(mutated_crossover_set))
        for i in temp_total:
            temp_data = total_features[:, i]
            try:
                ga_accuracy.append((i, evaulate(temp_data)))
            except ValueError:
                continue
        ga_accuracy = unique_accuracy(ga_accuracy)
        ga_accuracy.sort(key=lambda x: x[1])  # sort accuracy
        print(f"5 Best sets in generation {generation}: ")
        for a in ga_accuracy[-5:]:
            print(f"{a[0]} with accuracy: {a[1]}")
    # print(mutated_selected_set)
    generation += 1
final_fit(ga_accuracy[-1][0])
