from mlp import *
from arff import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import csv
from sklearn.preprocessing import OneHotEncoder

# Part 1
# DEBUGGING DATASET RESULTS
mat = Arff("datasets/linsep2nonorigin.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
widths = [2 * np.shape(data)[1]]
MLPClass = MLPClassifier(hidden_layer_widths=widths, lr=0.1, momentum=0.5, shuffle=False, deterministic=10)
MLPClass.fit(data,labels)
Accuracy = MLPClass.score(data, labels)

print("DEBUG DATASET")
print("Accuracy = [{:.2f}]".format(Accuracy))
print("Final Weights =",MLPClass.get_weights())
print()

# EVALUATION DATASET RESULTS
mat = Arff("datasets/data_banknote_authentication.arff",label_count=1)
np_mat = mat.data
data = np_mat[:,:-1]
labels = np_mat[:,-1].reshape(-1,1)
widths = [2 * np.shape(data)[1]]
MLPClass = MLPClassifier(hidden_layer_widths=widths, lr=0.1, momentum=0.5, shuffle=False, deterministic=10)
MLPClass.fit(data,labels)
Accuracy = MLPClass.score(data, labels)

print("EVALUATION DATASET")
print("Accuracy = [{:.2f}]".format(Accuracy))
print()

# Save weights to a csv file for the report
with open('evaluation.csv', mode='w') as eval_weight_file:
    eval_weight_writer = csv.writer(eval_weight_file, quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for W in MLPClass.get_weights():
        for i in range(np.shape(W)[0]):
            for j in range(np.shape(W)[1]):
                eval_weight_writer.writerow([W[i, j]])



# Part 2
# IRIS DATASET RESULTS
mat = Arff("datasets/iris.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
widths = [2 * np.shape(data)[1]]
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
MLPClass = MLPClassifier(hidden_layer_widths=widths, lr=0.1, momentum=0.0, shuffle=True, validation_size=0.15)
MLPClass.fit(X_train, y_train)
Accuracy = MLPClass.score(data, labels)

print("IRIS DATASET")
print("Accuracy = [{:.2f}]".format(Accuracy))
print()

x = range(MLPClass.n_epochs)
plt.plot(x, MLPClass.validation_mses, "-r", label="Validation MSE")
plt.plot(x, MLPClass.train_mses, "-b", label="Train MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Iris MSE Across Epochs")
plt.legend(loc="best")
plt.grid()
plt.show()
plt.plot(x, MLPClass.validation_accuracies, "-g", label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Across Epochs")
plt.legend(loc="best")
plt.grid()
plt.show()



# Part 3
# VOWEL DATASET RESULTS - BASELINE
mat = Arff("datasets/vowel.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
unique, counts = np.unique(labels, return_counts=True)
unique_map = dict(zip(unique, counts))
max_key = max(unique_map, key=unique_map.get)
max_class_count = unique_map[max_key]
n_instances = np.shape(data)[0]
baseline_accuracy = max_class_count / n_instances

print("VOWEL DATASET - BASELINE")
print("Accuracy = [{:.2f}]".format(baseline_accuracy))
print()

# VOWEL DATASET RESULTS - FINDING BEST LR
mat = Arff("datasets/vowel.arff",label_count=1)
data = mat.data[:,3:-1]
labels = mat.data[:,-1].reshape(-1,1)
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
widths = [2 * np.shape(data)[1]]
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
epochs_needed = []
mses = []
lrs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9, 2]

for lr in lrs:
    print('Current learning rate: ' + str(lr))
    curr_epochs_needed = []
    curr_mses = []

    for _ in range(3):
        MLPClass = MLPClassifier(hidden_layer_widths=widths, lr=lr, momentum=0.0, shuffle=False, validation_size=0.15)
        MLPClass.fit(X_train, y_train)
        curr_epochs_needed.append(MLPClass.bssf_n_epochs)
        curr_mses.append((MLPClass.bssf_ts_mse, MLPClass.bssf_vs_mse, MLPClass._calculate_mse(X_test, y_test, MLPClass.bssf_weights)))

    epochs_needed.append(sum(curr_epochs_needed) / 3)
    mses.append(curr_mses)

train_averages = []
val_averages = []
test_averages = []

for mse in mses:
    avg_ts_mse = 0
    avg_vs_mse = 0
    avg_test_mse = 0

    for tuple in mse:
        avg_ts_mse += tuple[0]
        avg_vs_mse += tuple[1]
        avg_test_mse += tuple[2]

    avg_ts_mse /= 3
    avg_vs_mse /= 3
    avg_test_mse /= 3

    train_averages.append(avg_ts_mse)
    val_averages.append(avg_vs_mse)
    test_averages.append(avg_test_mse)

x = np.arange(len(lrs))
width = 0.25
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, train_averages, width, label='Train MSE')
rects2 = ax.bar(x, val_averages, width, label='Validation MSE')
rects3 = ax.bar(x + width, test_averages, width, label='Test MSE')
ax.set_ylabel('MSE')
ax.set_title('Vowel MSE vs. Learning Rate')
ax.set_xticks(x)
ax.set_xticklabels(lrs)
ax.legend()
plt.xlabel('Learning Rate')
plt.show()

x = np.arange(len(lrs))
width = 0.75
fig, ax = plt.subplots()
rects1 = ax.bar(x, epochs_needed, width, label='Epochs')
ax.set_ylabel('Number of Epochs Needed to Stop')
ax.set_title('Number of Epochs Needed to Stop vs. Learning Rate')
ax.set_xticks(x)
ax.set_xticklabels(lrs)
ax.legend()
plt.xlabel('Learning Rate')
plt.show()



# Part 4
# VOWEL DATASET RESULTS - FINDING BEST NUMBER OF HIDDEN NODES
mat = Arff("datasets/vowel.arff",label_count=1)
data = mat.data[:,3:-1]
labels = mat.data[:,-1].reshape(-1,1)
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
mses = []
hidden_nodes = [1, 2, 4, 8, 16, 32, 64, 128]

for hidden_node in hidden_nodes:
    print('Current number of hidden nodes: ' + str(hidden_node))
    curr_mses = []

    for _ in range(3):
        MLPClass = MLPClassifier(hidden_layer_widths=[hidden_node], lr=0.1, momentum=0.0, shuffle=False, validation_size=0.15)
        MLPClass.fit(X_train, y_train)
        curr_mses.append((MLPClass.bssf_ts_mse, MLPClass.bssf_vs_mse, MLPClass._calculate_mse(X_test, y_test, MLPClass.bssf_weights)))

    mses.append(curr_mses)

train_averages = []
val_averages = []
test_averages = []

for mse in mses:
    avg_ts_mse = 0
    avg_vs_mse = 0
    avg_test_mse = 0

    for tuple in mse:
        avg_ts_mse += tuple[0]
        avg_vs_mse += tuple[1]
        avg_test_mse += tuple[2]

    avg_ts_mse /= 3
    avg_vs_mse /= 3
    avg_test_mse /= 3

    train_averages.append(avg_ts_mse)
    val_averages.append(avg_vs_mse)
    test_averages.append(avg_test_mse)

x = np.arange(len(hidden_nodes))
width = 0.25
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, train_averages, width, label='Train MSE')
rects2 = ax.bar(x, val_averages, width, label='Validation MSE')
rects3 = ax.bar(x + width, test_averages, width, label='Test MSE')
ax.set_ylabel('MSE')
ax.set_title('Vowel MSE vs. # Nodes in Hidden Layer')
ax.set_xticks(x)
ax.set_xticklabels(hidden_nodes)
ax.legend()
plt.xlabel('Hidden Layer Nodes')
plt.show()



# Part 5
# VOWEL DATASET RESULTS - FINDING BEST MOMENTUM
mat = Arff("datasets/vowel.arff",label_count=1)
data = mat.data[:,3:-1]
labels = mat.data[:,-1].reshape(-1,1)
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
epochs_needed = []
momentums = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for momentum in momentums:
    print('Current momentum: ' + str(momentum))
    curr_epochs_needed = []

    for _ in range(3):
        MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=momentum, shuffle=False, validation_size=0.15)
        MLPClass.fit(X_train, y_train)
        curr_epochs_needed.append(MLPClass.bssf_n_epochs)

    epochs_needed.append(sum(curr_epochs_needed) / 3)

plt.plot(momentums, epochs_needed)
plt.xlabel('Momentum')
plt.ylabel('Epochs')
plt.title('Vowel Epochs vs. Momentum')
plt.show()



# Part 6
# SKLEARN MLP ON VOWEL DATA
from sklearn import neural_network

mat = Arff("datasets/vowel.arff",label_count=1)
data = mat.data[:,3:-1]
labels = mat.data[:,-1].reshape(-1,1)
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 1 - default comparison")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64, 64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 2 - hidden layers/nodes")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='relu', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 3 - relu activation")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='tanh', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 4 - tanh activation")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.05, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.05, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 5 - smaller learning rate")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.5, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.5, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 6 - bigger learning rate")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0.0005)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 7 - small regularization")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0.05)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 8 - big regularization")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.2, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='logistic', momentum=0.2, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 9 - smaller momentum")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=True, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 10 - nesterov momentum")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.2)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=False, early_stopping=True, shuffle=False,
                            n_iter_no_change=30, alpha=0, validation_fraction=0.2)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 11 - bigger validation size for early stopping")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

MLPClass = MLPClassifier(hidden_layer_widths=[64], lr=0.1, momentum=0.9, shuffle=False, validation_size=0.1)

sklearn_mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, learning_rate_init=0.1, solver='sgd',
                            activation='logistic', momentum=0.9, nesterovs_momentum=False, early_stopping=False, shuffle=False,
                            n_iter_no_change=30, alpha=0)

MLPClass.fit(X_train, y_train)
MLPClass.weights = MLPClass.bssf_weights
sklearn_mlp.fit(X_train, y_train)

print("Run 12 - no early stopping")
print("My MLP test accuracy: " + str(MLPClass.score(X_test, y_test)))
print("My MLP number of epochs: " + str(MLPClass.n_epochs))
print("Sklearn's MLP test accuracy: " + str(sklearn_mlp.score(X_test, y_test)))
print("Sklearn's MLP number of epochs: " + str(sklearn_mlp.n_iter_))
print()

# # I DO NOT RECOMMEND RUNNING THIS CODE, IT TAKES ABOUT 20 MINUTES TO RUN
# # Custom dataset and grid search
# mat = Arff("datasets/breast-w.arff",label_count=1)
# mat.data = mat.data[~np.isnan(mat.data).any(axis=1)]
# data = mat.data[:,:-1]
# labels = mat.data[:,-1].reshape(-1,1)
#
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
#
# from sklearn.model_selection import GridSearchCV
# param_grid = {'hidden_layer_sizes': [(64), (32, 64), (64,64), (32,64,64)],
#               'max_iter': [1000],
#               'learning_rate_init': [0.1, 0.2],
#               'solver': ['sgd'],
#               'activation': ['relu', 'tanh', 'logistic'],
#               'momentum': [0.7, 0.8, 0.9],
#               'nesterovs_momentum': [True, False],
#               'early_stopping': [True, False],
#               'shuffle': [True, False],
#               'n_iter_no_change': [30]}
#
# grid_search = GridSearchCV(neural_network.MLPClassifier(), param_grid, cv=6, return_train_score=True)
# grid_search.fit(X_train, y_train)
#
# print("Best accuracy: {:.5f}".format(grid_search.best_score_))
# print("Best hyperparameters: {}".format(grid_search.best_params_))

