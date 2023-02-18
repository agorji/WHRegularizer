from rf_fourier.node import Node
from rf_fourier.fourier import Fourier
import json
from collections import deque
import numpy as np
from tqdm import tqdm
from decimal import Decimal

class RFFourierExtractor:
    def __init__(self, rf_model):
        self.node_stack = []
        self.fourier_transform = Fourier.zero()

        for decision_tree_regressor in tqdm(rf_model.estimators_, total=len(rf_model.estimators_)):
            tree = RFFourierExtractor.build_tree_from_sklearn(decision_tree_regressor)
            self.fourier_transform += RFFourierExtractor.get_fourier(tree)
        self.fourier_transform /= rf_model.n_estimators

        print(f"Sparsity = {self.get_fourier_transform().get_sparsity()}")
        self.sparsity = self.get_fourier_transform().get_sparsity()
        self.sampling_complexity = 0
        self.use_cache = False
        self.cache = []
        self.cache_read_index = 0


    def clear_cache(self):
        self.cache = []
        self.cache_read_index = 0

    def reset_sampling_complexity(self):
        self.sampling_complexity = 0

    def get_sampling_complexity(self):
        return self.sampling_complexity

    def __getitem__(self, item):
        self.sampling_complexity += 1
        if self.use_cache:
            value = self.cache[self.cache_read_index]
            self.cache_read_index += 1
            return value
        else:
            self.cache.append(self.regr.predict(np.reshape(item, (1, -1))).item())
            return self.cache[-1]

    def __call__(self, item):
        return self.__getitem__(self, item)

    def get_fourier_transform(self):
        return self.fourier_transform

    @staticmethod
    def build_tree_from_sklearn(decision_tree_regressor):
        sk_tree = decision_tree_regressor.tree_
        root = Node()
        sk_queue = deque([[0, root]])
        tree_node_stack = []
        while sk_queue:
            sk_node_id, current_node = sk_queue.popleft()
            tree_node_stack.append(current_node)
            if sk_tree.children_left[sk_node_id] != sk_tree.children_right[sk_node_id]:
                # We have a split node
                current_node.label = sk_tree.feature[sk_node_id]
                current_node.left, current_node.right = Node(depth=current_node.depth), Node(depth=current_node.depth)
                sk_queue.append([sk_tree.children_left[sk_node_id], current_node.left])
                sk_queue.append([sk_tree.children_right[sk_node_id], current_node.right])
            else:
                # We have a leaf node
                current_node.value = sk_tree.value[sk_node_id][0][0]
        return tree_node_stack

    @staticmethod
    def get_fourier(tree_node_stack):
        for node in reversed(tree_node_stack):
            if node.is_leaf():
                node.computed_fourier = Fourier({frozenset(): Decimal(node.value)})
            else:
                label = node.label
                fourier_left = node.left.computed_fourier
                fourier_right = node.right.computed_fourier
                new_series = {}

                for key in fourier_left.series:
                    new_series[key] = new_series.get(key, Decimal()) + (fourier_left.series[key] / Decimal(2))
                    new_series[key.union([label])] = new_series.get(key.union([label]), Decimal()) + (fourier_left.series[key] /  Decimal(2))

                for key in fourier_right.series:
                    new_series[key] = new_series.get(key, Decimal()) + (fourier_right.series[key] /  Decimal(2))
                    new_series[key.union([label])] = new_series.get(key.union([label]), Decimal()) - (fourier_right.series[key] /  Decimal(2))

                node.computed_fourier = Fourier(new_series)
        
        return tree_node_stack[0].computed_fourier