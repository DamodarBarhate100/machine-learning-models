import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_idx = None, threshold=None, info_gain = None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right  = right

        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100):
        self.min_samples_split  = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def build_tree(self, dataset, current_depth = 0):
        X, y = dataset[:, :-1], dataset[:,-1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and current_depth<=self.max_depth:
            best_split  = self.best_split(dataset, n_features )

            if best_split["info_gain"]>0:
                left_node = self.build_tree(best_split['left_dataset'], current_depth + 1)
                right_node = self.build_tree(best_split['right_dataset'], current_depth + 1)

                return Node(best_split['feature_idx'], best_split['threshold'], best_split["info_gain"],left_node, right_node)

        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    def best_split(self, dataset, n_features):
        best_split = {"feature_idx":None, "threshold":None, "info_gain":-1, "left_dataset":None, "right_dataset":None, }


        for i in range(n_features):
            feature_values =  dataset[:, i]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_dataset, right_dataset = self.split(dataset,i, threshold)

                if len(left_dataset) and len(right_dataset):
                    parent_y, left_y, right_y = dataset[:,-1], left_dataset[:,-1], right_dataset[:,-1]

                    info_gain = self.information_gain(parent_y, left_y, right_y)

                    if info_gain > best_split['info_gain']: 
                        best_split['feature_idx'] = i
                        best_split['threshold'] = threshold
                        best_split['info_gain'] = info_gain
                        best_split['left_dataset'] =   left_dataset
                        best_split['right_dataset'] =   right_dataset
        return best_split

    def split(self,dataset, feature_idx, threshold):
        left_dataset = dataset[dataset[:, feature_idx] <= threshold]
        right_dataset = dataset[dataset[:, feature_idx] > threshold]
        return left_dataset, right_dataset

    def information_gain(self, parent_y, left_y, right_y):
        left_weight = len(left_y)/len(parent_y)
        right_weight = len(right_y)/len(parent_y)

        gain = self.entropy(parent_y) - (left_weight * self.entropy(left_y) + right_weight * self.entropy(right_y))

        return gain

    def entropy(self, y):
        unique_class, counts = np.unique(y,return_counts=True)
        probabilities = counts/len(y)
        probabilities = probabilities[probabilities>0]
        entropy_ = -np.sum(probabilities * np.log2(probabilities))
        return entropy_

    def fit(self,X,y):
        dataset = np.concatenate([X,y.reshape(-1,1)],axis=1)
        self.root = self.build_tree(dataset)


    def predict(self, X):
        predictions = [self.predict_class(row,self.root) for row in X]
        return np.array(predictions, dtype=int)

    def predict_class(self, row, node):
        if node.value != None:
            return node.value 

        feature_val = row[node.feature_idx]
        if feature_val <= node.threshold:
            return self.predict_class(row, node.left)
        else:
            return self.predict_class(row, node.right)



                             




