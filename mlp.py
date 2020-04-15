import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=None, validation_size=0.15):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle if deterministic is None else False
        self.deterministic = deterministic
        self.validation_size = validation_size if deterministic is None else 0
        self.weights = []


    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # Get the number of features and targets
        self.num_features = np.shape(X)[1]
        self.num_targets = np.shape(y)[1]

        # Use numpy to add a bias column (axis=1) to the input
        X = np.concatenate((X, np.ones((np.shape(X)[0], 1))), axis=1)

        self.train_mses = []
        self.train_accuracies = []

        # Get a training set and validation set if we need to
        if self.validation_size > 0:
            self.validation_mses = []
            self.validation_accuracies = []
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.validation_size)
            # Set up variables for the best accuracy so far and the weights corresponding to that accuracy
            self.bssf_accuracy = 0
            self.bssf_mse = 1
            self.bssf_weights = []
            self.bssf_n_epochs = 0
            self.bssf_ts_mse = 0
            self.bssf_vs_mse = 0
            n_epochs_since_change = 0

        else:
            X_train = X
            y_train = y

        # Initialize the weights
        self.weights = self.initialize_weights() if initial_weights is None else initial_weights

        # Lists for storing information needed for back-propagation
        self.outputs = [0] * (len(self.weights) + 1)
        self.deltas = [0] * (len(self.weights) + 1)
        self.weight_updates = [0] * len(self.weights)
        self.previous_updates = [0] * len(self.weights)

        # Set stop flag for our training loop and initialize the number of epochs we will run
        self.n_epochs = 0

        # Run through epochs until our stopping criteria is met
        while True:
            # Shuffle the data at the start of each epoch (if we are not running deterministically)
            if self.shuffle:
                X_train, y_train = self._shuffle_data(X_train, y_train)

            for i in range(np.shape(X_train)[0]):
                self._train([X_train[i]], y_train[i])

            # Calculate the accuracy at the end of the epoch
            self.train_accuracies.append(self.score(X_train, y_train))
            self.train_mses.append(self._calculate_mse(X_train, y_train))

            # Increment the number of epochs
            self.n_epochs += 1

            # If we're using a validation set as the stopping criteria
            if self.validation_size > 0:
                validation_accuracy = self.score(X_valid, y_valid)
                validation_mse = self._calculate_mse(X_valid, y_valid)

                self.validation_accuracies.append(validation_accuracy)
                self.validation_mses.append(self._calculate_mse(X_valid, y_valid))

                # If our validation mse improved, update the best solution so far
                if validation_mse < self.bssf_mse:
                    self.bssf_mse = validation_mse
                    self.bssf_accuracy = validation_accuracy
                    self.bssf_weights = self.weights
                    self.bssf_n_epochs = self.n_epochs
                    self.bssf_ts_mse = self._calculate_mse(X_train, y_train)
                    self.bssf_vs_mse = self._calculate_mse(X_valid, y_valid)
                    n_epochs_since_change = 0

                else:
                    n_epochs_since_change += 1

                # If we've gone 30 epochs without an improvement in validation accuracy, stop
                if n_epochs_since_change == 30 and self.bssf_accuracy > 0:
                    break

                # If we ever run too many epochs, just stop
                elif self.n_epochs == 1000:
                    break

            # If we are running deterministically, check if we've reached the set number of epochs
            if self.deterministic is not None and self.n_epochs == self.deterministic:
                break

        return self

    def _train(self, X, y):
        self.predict(X)

        for i in range(len(self.weights), -1, -1):
            if i == len(self.weights):
                self.deltas[i] = (y - self.outputs[i]) * self.outputs[i] * (1 - self.outputs[i])

            else:
                self.deltas[i] = np.dot(self.deltas[i + 1], np.transpose(self.weights[i])[:, :-1]) * (self.outputs[i][:, :-1] * (1 - self.outputs[i][:, :-1]))

                if i == len(self.weights) - 1:
                    weight_update = np.dot(np.transpose(self.outputs[i]), self.deltas[i + 1]) * self.lr + \
                                    (self.momentum * self.previous_updates[i])

                else:
                    weight_update = np.dot(np.transpose(self.outputs[i]), self.deltas[i + 1]) * self.lr + \
                                    (self.momentum * self.previous_updates[i])

                self.weight_updates[i] = weight_update
                self.previous_updates[i] = weight_update

        for i in range(len(self.weights)):
            self.weights[i] += self.weight_updates[i]

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        # If there is no input bias feature, add it
        if np.shape(X)[1] != self.num_features + 1:
            X = np.concatenate((X, np.ones((np.shape(X)[0], 1))), axis=1)

        self.outputs[0] = np.array(X)

        for i in range(len(self.weights)):
            net = np.dot(self.outputs[i], self.weights[i])
            output = 1 / (1 + np.exp(-net))

            # Add bias output
            if i < len(self.weights) - 1:
                output = np.concatenate((output, np.ones((np.shape(output)[0], 1))), axis=1)

            self.outputs[i + 1] = output

        return self.outputs[-1]

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        weights = []

        # If we are running deterministically, we want all the weights to be initialized to 0
        if self.deterministic is not None:
            low = 0
            high = 0

        # Otherwise, use the range given in the book
        else:
            low = -1 / (self.num_features + 1)
            high = 1 / (self.num_features + 1)

        # Initialize the weights for the input the layer
        # Use np.random.uniform to generate the weights
        weights.append(np.random.uniform(low=low, high=high, size=(self.num_features + 1, self.hidden_layer_widths[0])))

        # Initialize the weights for the hidden layers
        for i in range(len(self.hidden_layer_widths)):
            # Column dimension of num_targets for the hidden state that goes to the output layer (the last hidden state)
            if i == len(self.hidden_layer_widths) - 1:
                size = (self.hidden_layer_widths[i] + 1, self.num_targets)

            else:
                size = (self.hidden_layer_widths[i] + 1, self.hidden_layer_widths[i + 1])

            # If we are running deterministically, we want all the weights to be initialized to 0
            if self.deterministic is not None:
                low = 0
                high = 0

            # Otherwise, use the range given in the book
            else:
                low = -1 / (self.hidden_layer_widths[i] + 1)
                high = 1 / (self.hidden_layer_widths[i] + 1)

            # Use np.random.uniform to generate the weights
            weights.append(np.random.uniform(low=low, high=high, size=size))

        return weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        # Use our predict method
        output = self.predict(X)

        # Find the total number of correct predictions and the total number of data instances
        num_correct = np.sum(np.all(np.rint(output) == np.array(y), axis=1))
        num_instances = np.shape(X)[0]

        # Calculate and return the accuracy
        return num_correct / num_instances

    def _calculate_mse(self, X, y, weights=None):
        tmp = self.weights

        if weights is not None:
            self.weights = weights

        # Use our predict method
        output = self.predict(X)

        num_instances = np.shape(X)[0]

        self.weights = tmp

        return np.sum((y - output)**2) / num_instances

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        # Use numpy to create a random permutation of the numbers 0 to the number of data instances
        p = np.random.permutation(np.shape(X)[0])

        # Use the permutation to return shuffled X and y data
        # Using the same permutation for both X and y ensures that they are shuffled in unison
        return X[p], y[p]

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
