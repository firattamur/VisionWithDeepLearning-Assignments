from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L1 and L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
            consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
            y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0, distfn='L2'):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
            of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
            between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            if distfn == 'L2':
                dists = self.compute_L2_distances_no_loops(X)
            else:
                dists = self.compute_L1_distances_no_loops(X)
        elif num_loops == 1:
            if distfn == 'L2':
                dists = self.compute_L2_distances_one_loop(X)
            else:
                dists = self.compute_L1_distances_one_loop(X)
        elif num_loops == 2:
            if distfn == 'L2':
                dists = self.compute_L2_distances_two_loops(X)
            else:
                dists = self.compute_L1_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_L2_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            is the Euclidean distance between the ith test point and the jth training
            point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):

            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                # l2 distance
                # sum((x - y) ** 2) ** (1/2)

                # get ith test data
                test_i = X[i]

                # get jth train data
                train_j = self.X_train[j]

                # subtract X - Y
                subtracted = test_i - train_j

                # square of subtraction
                squared = np.power(subtracted, 2)

                # sum of squares values
                sum_squared = np.sum(squared)

                # square root
                square_root = np.power(sum_squared, 0.5)

                # set distance
                dists[i][j] = square_root

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L2_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_L2_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # l2 distance
            # sum((x - y) ** 2) ** (1/2)

            # get ith test data
            # shape = (1, features)
            test_i = X[i]

            # subtract test_i from all train data matrix
            # shape = (train_num, features)
            subtracted = self.X_train - test_i

            # square of subtraction
            # shape = (train_num, features)
            squared = np.power(subtracted, 2)

            # sum of squares values
            # shape = (train_num, 1)
            sum_squared = np.sum(squared, axis=1)

            # square root
            # shape = (train_num, 1)
            square_root = np.power(sum_squared, 0.5)

            # reshape (train_num, 1) to (1, train_num)
            square_root_reshaped = square_root.reshape(1, -1)

            # set distance
            # shape = (test_num, train_num)
            dists[i] = square_root_reshaped

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L2_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_L2_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # l2 distance
        # sum((x - y) ** 2) ** (1/2)
        # efficient way 
        # num_feat = self.X_train.shape[1]
    
        # X_train_tranpose = self.X_train.T
        # X_test_traponse  = X.T
        
        # # repeate X_train cols by number of X_test
        # X_train_repeated = np.repeat(X_train_tranpose, num_test, axis=1)

        # # repeated whole X_test matrix
        # X_test_repeated = np.tile(X_test_traponse, num_train)

        # # substract X_test from X_train
        # sub = X_train_repeated - X_test_repeated

        # # square substraction
        # sub_squared = np.power(sub, 2)

        # # sum of squareds
        # sub_squared_sum = np.sum(sub_squared, axis=0)

        # # square root of sum
        # sub_squared_sum_root = np.power(sub_squared_sum, 0.5)

        # # reshape into (num_train, num_feat)
        # sub_squared_sum_root = sub_squared_sum_root.reshape(num_train, num_feat)

        # # take transpose to react distances
        # dists = sub_squared_sum_root.T

        # because column of matrix repeats the first solution was slow.
        # searching for faster answer we found that:
        # (x - y) ** 2 = x**2 - 2xy + y**2

        # X_train_square = np.sum(np.square(self.X_train), axis=1)
        # X_test_square  = np.sum(np.square(X), axis=1)
        # two_X_train_X_test = 2 * (X @ self.X_train.T)

        # dists = X_test_square.reshape(-1, 1) - two_X_train_X_test + X_train_square.reshape(1, -1)

        dists = np.sqrt(np.sum(X**2, axis=1).reshape(num_test, 1) + np.sum(self.X_train**2, axis=1) - 2 * X.dot(self.X_train.T))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L1_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            is the Euclidean distance between the ith test point and the jth training
            point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l1 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                # l1 distance
                # abs(x - y)

                # get ith test data
                test_i = X[i]

                # get jth train data
                train_j = self.X_train[j]

                # subtract X - Y
                subtracted = test_i - train_j

                # absulate of subtraction
                absolute = np.abs(subtracted)

                # sum of absoluate values
                sum_absolute = np.sum(absolute)

                # set distance
                dists[i][j] = sum_absolute


                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L1_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_L1_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l1 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                        # l2 distance
            # sum((x - y) ** 2) ** (1/2)

            # get ith test data
            # shape = (1, features)
            test_i = X[i]

            # subtract test_i from all train data matrix
            # shape = (train_num, features)
            subtracted = self.X_train - test_i

            # square of subtraction
            # shape = (train_num, features)
            absolute = np.abs(subtracted)

            # sum of squares values
            # shape = (train_num, 1)
            sum_absolute = np.sum(absolute, axis=1)

            # square root
            # shape = (train_num, 1)
            sum_absolute_reshaped = sum_absolute.reshape(1, -1)

            # set distance
            # shape = (test_num, train_num)
            dists[i] = sum_absolute_reshaped

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L1_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_L1_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l1 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l1 distance using broadcast operations     #
        #                                                                       #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # working but really slow in high in dimensions because of a lot of repeating
        # dists = np.sum(np.abs(np.repeat(self.X_train.T, X.shape[0], axis=1) - np.tile(X.T, self.X_train.shape[0])), axis=0).reshape(self.X_train.shape[0], -1).T

        # version 2 faster than first one
        # dists = np.abs(self.X_train.flatten() - np.tile(X, self.X_train.shape[0])).reshape(-1, self.X_train.shape[0] * X.shape[0], self.X_train.shape[1]).sum(axis=2).reshape(X.shape[0], self.X_train.shape[0])
        # dists = np.abs(X.flatten() - np.tile(self.X_train, X.shape[0])).reshape(-1, self.X_train.shape[0] * X.shape[0], self.X_train.shape[1]).sum(axis=2).reshape(self.X_train.shape[0], X.shape[0]).T
        
        # fastest one
        dists = np.abs(self.X_train[:, np.newaxis] - X).sum(axis=-1).T
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
            gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
            test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # sorts distances and return indices
            closests = np.argsort(dists[i, :])

            # get k closest data point indices
            k_closest_indices = closests[:k]

            # get labels fromt closest train points
            closest_y = self.y_train[k_closest_indices] 

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            y_pred[i] = np.bincount(closest_y).argmax()

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
