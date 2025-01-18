from math_functions import sqrt

class ScaleData:
    def __init__(self, method = "minmax"):
        self.method = method
        self.min = None
        self.max = None
        self.means = None
        self.std_devs = None
    
    def fit(self, X_train):
        # z score scaling
        if self.method == "standard":
            # calc mean
            self.means = []
            for feature in zip(*X_train):
                self.means.append(sum(feature) / len(feature))
            
            # calc std
            self.std_devs = []
            for feature, mean in zip(zip(*X_train), self.means):
                x_mean_sqr = 0
                for x in feature:
                    x_mean_sqr += (x-mean)**2
                variance = x_mean_sqr / len(feature)
                self.std_devs.append(sqrt(variance))

        # min max scaling
        if self.method == "minmax":
            self.max = []
            self.min = []
            for feature in zip(*X_train):
                self.max.append(max(feature))
                self.min.append(min(feature))

    def transform(self, X_train):
        # z score
        if self.method == "standard":
            transformed_data = []
            for row in X_train:
                transformed_row = []
                for x, mean, std in zip(row, self.means, self.std_devs):
                    transformed_row.append((x-mean)/ (std + 1e-9))
                transformed_data.append(transformed_row)
            return transformed_data

        # minmax
        if self.method == "minmax":
            transformed_data = []
            for row in X_train:
                transformed_row = []
                for x, maxVal, minVal in zip(row, self.max, self.min):
                    transformed_row.append((x-minVal) / (maxVal - minVal + 1e-9))
                transformed_data.append(transformed_row)
            return transformed_data