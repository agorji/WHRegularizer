import numpy as np
from rf_fourier.fourier import Fourier
from decimal import Decimal, getcontext
"""
The Node class is for building decision trees recursively
Each instance of this class is node (either leaf of non-leaf)
"""

getcontext().prec = 20

class Node:
    def __init__(self, depth=0, value=None, label=None, left=None, right=None):
        # self.value is the real number in the node  is a leaf and None if it is a split_node
        self.value = value
        # self.label is the features assigned to the node if it is a split_node and None if it is a leaf
        self.label = label
        # self.left is the left subtree (can be None)
        self.left = left
        # self.right is the right subtree (can be None)
        self.right = right
        # self.depth is the depth of the node
        self.depth = depth

        self.computed_fourier = None
    


    def __str__(self):
        return "label:" + str(self.label) + " value:" + str(self.value) + " left:" + str(self.left) + "right:" + str(self.right)

    """
    # Returns a Fourier series (instance of the Fourier class) that defines the same function as the decision tree 
    defined by this object
    """
    def get_fourier(self):
        if self.is_leaf():
            return Fourier({frozenset(): Decimal(self.value)})

        else: # it is a variable node
            label = self.label
            fourier_left = self.left.get_fourier()
            fourier_right = self.right.get_fourier()
            new_series = {}

            for key in fourier_left.series:
                new_series[key] = new_series.get(key, Decimal()) + (fourier_left.series[key] / Decimal(2))
                new_series[key.union([label])] = new_series.get(key.union([label]), Decimal()) + (fourier_left.series[key] /  Decimal(2))

            for key in fourier_right.series:
                new_series[key] = new_series.get(key, Decimal()) + (fourier_right.series[key] /  Decimal(2))
                new_series[key.union([label])] = new_series.get(key.union([label]), Decimal()) - (fourier_right.series[key] /  Decimal(2))

            return Fourier(new_series)

    """
    Recursively compute depth of the tree
    """
    def get_depth(self):
        if self.is_leaf():
            return Decimal()
        else:
            return max(self.left.get_depth(), self.right.get_depth()) + 1
    """
    Recursively compute node count of the tree including leaves
    """
    def get_node_count(self):
        if self.is_leaf():
            return 1
        else:
            return 1 + self.left.get_node_count() + self.right.get_node_count()

    """
    Recursviely compute number of leaves in the tree
    """
    def get_leaf_count(self):
        if self.is_leaf():
            return 1
        else:
            return self.left.get_leaf_count() + self.right.get_leaf_count()

    """
    Get tree prediction for input x
    """
    def predict(self, X):
        predictions = []
        for row in X:
            predictions.append(self.__getitem__(row))
        return np.array(predictions)

    """
    Returns true if node is a leaf
    """
    def is_leaf(self):
        return self.value != None

    """
    If that coordinate has value 0 we go left and if it has value 1 we go right. 
    The value of the function is the real number in the leaf we reach.
    """
    def __getitem__(self, argument):
        if self.is_leaf():
            return self.value
        else:
            if argument[self.label] == 0:
                return self.left[argument]
            elif argument[self.label] == 1:
                return self.right[argument]
            else:
                raise Exception("argument can only contain 0s and 1s")

if __name__ == "__main__":
    #Leaf nodes
    A = Node(value=10)
    B = Node(value=20)
    C = Node (value=30)
    D = Node(value=40)
    # Variable nodes
    x1 = Node(label=1, left=A, right=B )
    x2 = Node(label=2, left=C, right=D)
    x0 = Node(label=0, left=x1, right=x2)
