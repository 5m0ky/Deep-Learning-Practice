import numpy as np

class findMeanSquaredError(object):
    def __init__(self):
        self.y_predicted = np.array([1, 1, 0, 0, 1])
        self.y_true = np.array([0.30, 0.7, 1, 0, 0.5])

    def mse(self):
        self.total_error = 0
        self.mean_s_error = 0

        for i, j in zip(self.y_true, self.y_predicted):
            self.total_error = (i - j) ** 2
        print(f"The total error is: {self.total_error}")
        
        self.mean_s_error = self.total_error / len(self.y_true)
        print(f"The mean squared error is: {self.mean_s_error}")
        

m = findMeanSquaredError()
m.mse()
