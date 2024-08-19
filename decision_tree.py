class MyDecisionTree:
    def __init__(self, criterion='random', max_depth=10, min_sample_split=2) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(y) < self.min_sample_split:
            return LeafNode(self._get_leaf_value(y))

        feature_index, threshold = self._choose_feature(X, y)

        if feature_index is None or threshold is None:
            return LeafNode(self._get_leaf_value(y))
        min_value = np.min(X.iloc[:, feature_index])
        max_value = np.max(X.iloc[:, feature_index])

        left_indices = X.iloc[:, feature_index] < threshold
        right_indices = X.iloc[:, feature_index] >= threshold

        if left_indices.sum() == 0 or right_indices.sum() == 0:
            return LeafNode(self._get_leaf_value(y))

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return SplitNode(feature_index, threshold, left_tree, right_tree)

    def _choose_feature(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = float('inf')

        for feature_index in range(X.shape[1]):
            feature_values = X.iloc[:, feature_index].sort_values().unique()
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2
            for threshold in thresholds:
                gini = self._calculate_gini(X, y, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def _calculate_gini(self, X, y, feature_index, threshold):
        left_indices = X.iloc[:, feature_index] < threshold
        right_indices = X.iloc[:, feature_index] >= threshold

        n = len(y)
        n_left = left_indices.sum()
        n_right = right_indices.sum()

        if n_right == 0 or n_left == 0: # Cas ou division separe mal donc pas efficace
            return float('inf') # On retourne inf pour definir que cette division est tres mauvaise

        left_gini = 1 - sum((np.sum(y[left_indices] == x) / n_left) ** 2 for x in np.unique(y))
        right_gini = 1 - sum((np.sum(y[right_indices] == x) / n_right) ** 2 for x in np.unique(y))

        weighted_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        return weighted_gini
    
    def _get_leaf_value(self, y):
        if y.empty:
            return print(y)
        else:
            return y.mode().iloc[0]

    def predict(self, X):
        return np.array(X.apply(self._predict_one, axis=1))

    def _predict_one(self, x):
        node = self.tree
        while isinstance(node, SplitNode):
            feature_value = x.iloc[node.feature_index]
            if not isinstance(feature_value, (int, float)):
                raise ValueError(f"Feature value {feature_value} at index {node.feature_index} is not numeric")
            if feature_value <= node.threshold:
                if node.left == None:
                    node = node.right
                else:
                    node = node.left
            else:
                if node.right == None:
                    node = node.left
                else:
                    node = node.right
        return node.value

class SplitNode:
    def __init__(self, feature_index, threshold, left, right) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

class LeafNode:
    def __init__(self, value) -> None:
        self.value = value
