from math_functions import euclidean_distance, manhatten_distance, minkowski_distance, kronecker_delta

class KNearestNeighbor:
    def __init__(self, k=3, distance_function = None, scale_data = False):
        self.k = k
        self.distance_function = distance_function or euclidean_distance
        self.scale_data = scale_data
        self.data = []

    # The KNN does not optimize a loss function like other ML models. When a prediction is made, the KNN simply compares the input with the
    # training data it has stored. 
    def fit(self, X, y):
        if X is None or y is None:
            raise ValueError("Input is none!")
        
        feature_length = len(X[0])
        if any(len(features) != feature_length for features in X):
            raise ValueError("All training points must have the same dimension!")
        
        if len(X) != len(y):
            raise ValueError("Features and labels must have the same length!")
        
        if any(not isinstance(features, list) for features in X):
            raise ValueError("Each element in X must be a list of features!")
        
        if any(element is None for features in X for element in features):
            raise ValueError("Features contain None values!")
    
        if any(label is None for label in y):
            raise ValueError("Labels contain None values!")
        
        for features, label in zip(X, y):
            self.data.append((features, label))

    def majority_vote(self, labels):
        d = {}
        unique_labels = set(labels)

        for l in unique_labels:
            d[l] = 0
            for x in labels:
                d[l] += kronecker_delta(l, x)
                
        max_value = max(d.values())

        c = []
        for key, val in d.items():
            if val == max_value:
                c.append(key)

        if len(c) > 1:
            random_index = hash(c[0]) % len(c)
            return c[random_index]
        else:
            return c[0]

    # The distance in the feature space between the query instance and each instance in memory is computed.
    def predict(self, X):
        predictions = []
        for test_point in X:
            # calc distance of test point to each point in training
            distances = []
            for i in range(len(self.data)):
                # safe distance and label as tupel to distances list
                print(f"Test point: {test_point}, Dimension: {len(test_point)}")
                print(f"Training point: {self.data[i][0]}, Dimension: {len(self.data[i][0])}")
                distances.append((self.distance_function(test_point, self.data[i][0]), self.data[i][1]))

            # sort based on first value (distances)
            distances.sort(key= lambda d: d[0])
            k_nearest = distances[:self.k]
            k_nearest_labels = []
            for _, label in k_nearest:
                k_nearest_labels.append(label)

            predictions.append(self.majority_vote(k_nearest_labels))

        return predictions