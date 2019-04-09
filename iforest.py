
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.ntrees = n_trees
        self.sample = sample_size
        self.limit = int(np.ceil(np.log2(self.sample)))
        self.c = self.c_factor(self.sample)


    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.size = len(X)
        self.trees = []
        for i in range(self.ntrees):
            ix = np.random.randint(0, self.size, self.sample)
            X_p = X[ix]
            itree = IsolationTree(self.limit)
            itree.fit(X_p, improved)
            self.trees.append(itree)
        return self


    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X. Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        path_list = []
        for x in X:
            path_x = [self.path(x, itree.root, 0) for itree in self.trees]
            path_list.append(path_x)
        return np.array(path_list).mean(1)


    def c_factor(self,n):
        if n > 2:
            return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
        elif n==2:
            return 1
        else:
            return 0


    def path(self, x, tree, current_height):
        if isinstance(tree, ExNode):
            return current_height + self.c_factor(tree.size)
        a = tree.split_attr
        if x[a] < tree.split_point:
            return self.path(x, tree.left, current_height+1)
        else:
            return self.path(x, tree.right, current_height+1)


    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.pathlength = self.path_length(X)
        return np.array([2**(-l/self.c) for l in self.pathlength])


    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.array([1 if s > threshold else 0 for s in scores])


    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X),threshold)


class ExNode:
    def __init__(self, X):
        self.size = len(X)


class InNode:
    def __init__(self, X, left, right, splitattr, splitpoint):
        self.left = left
        self.right = right
        self.split_attr = splitattr
        self.split_point = splitpoint
        self.size = len(X)


class IsolationTree:
    def __init__(self, height_limit):
        self.limit = height_limit
        self.n_nodes = 1

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.dim = np.shape(X)[1]
        if improved is False:
            self.root = self.make_tree(X,0)
        else:
            self.root = self.make_tree_new(X,0)
        return self.root

    def make_tree(self, X:np.ndarray, current_height):
        if (current_height >= self.limit) or (len(X)<=1):
            return ExNode(X)
        else:
            q = np.random.randint(0,self.dim)
            pmin = X[:,q].min()
            pmax = X[:,q].max()
            if (pmin == pmax):
                return ExNode(X)
            else:
                p = np.random.uniform(pmin, pmax)
                self.n_nodes += 2
                w = np.where(X[:,q]< p, True, False)
            return InNode(X, self.make_tree(X[w], current_height+1),
            self.make_tree(X[~w], current_height+1), q, p)

    def make_tree_new(self, X:np.ndarray, current_height):
        if (current_height >= self.limit) or (len(X)<=1):
            return ExNode(X)
        else:
            q = np.random.randint(0,self.dim)
            pmin = X[:,q].min()
            pmax = X[:,q].max()
            if (pmin == pmax):
                return ExNode(X)
            else:
                self.n_nodes += 2
                if X[:,q].mean() > np.median(X[:,q]):
                    p = np.random.uniform(np.percentile(X[:,q],90),pmax)
                else:
                    p = np.random.uniform(pmin, np.percentile(X[:,q],10))
                w = np.where(X[:,q]< p, True, False)
            return InNode(X, self.make_tree(X[w], current_height+1),
            self.make_tree(X[~w], current_height+1), q, p)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0
    pred = np.array([1 if s > threshold else 0 for s in scores])
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    while tpr < desired_TPR:
        threshold = threshold - 0.01
        pred = np.array([1 if s > threshold else 0 for s in scores])
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
    return threshold, fpr
