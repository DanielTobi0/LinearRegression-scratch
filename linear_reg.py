class LinearRegression:
    def __init__(self, feature, target):
        """
        A simple linear regression without any inbuilt functions.

        y = intercept + slope * feature (formular for linear regression)
        slope = pearson_correlation * (std of target /  std of feature)
        intercept = mean of target - (slope * mean of feature)

        pearson_correlation = sum( (deviation of feature) * (deviation of target) )
                                    /   square_root( (squared deviation of feature) * (squared deviation of target) )

        """
        self.feature = feature
        self.target = target

        self.feature_mean = sum(self.feature) / len(self.feature)  # mean of feature
        self.target_mean = sum(self.target) / len(self.target)  # mean of target

        self.feature_deviation_list = [x - self.feature_mean for x in self.feature]  # (feature - feature bar)
        self.target_deviation_list = [x - self.target_mean for x in self.target]  # (target - target bar)
        self.product_deviation = sum([self.feature_deviation_list[i] * self.target_deviation_list[i] for i in
                                      range(len(self.feature_deviation_list))])

        self.feature_deviation_squared = sum(
            [x ** 2 for x in self.feature_deviation_list])  # (feature - feature bar) ** 2
        self.target_deviation_squared = sum(
            [x ** 2 for x in self.target_deviation_list])  # (target - target bar) ** 2

        self.std_feature = (self.feature_deviation_squared / (len(self.feature) - 1)) ** 0.5  # STD of feature
        self.std_target = (self.target_deviation_squared / (len(self.target) - 1)) ** 0.5  # STD of target

        self.pearson_correlation = self.product_deviation / (
                (self.feature_deviation_squared) * (self.target_deviation_squared)) ** 0.5
        self.slope = self.pearson_correlation * (self.std_target / self.std_feature)
        self.intercept = self.target_mean - (self.slope * self.feature_mean)

    def predict(self, x):
        return self.intercept + self.slope * x


feature = [17, 13, 12, 15, 16, 14, 16, 16, 18, 19]
target = [94, 73, 59, 80, 93, 85, 66, 79, 77, 91]
linear = LinearRegression(feature, target)
print(linear.predict(15))
