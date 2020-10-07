import numpy as np

y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])
epsilon = 1e-15

def findMeanAbsoluteError(y_true, y_predicted):
    total_error = 0
    for i, j in zip(y_predicted, y_true):
        total_error = abs(i - j)
    print(f"Total Error: {total_error}")

    mean_absolute_error = total_error / len(y_true)

    print(f"Mean Absolute Error: {mean_absolute_error}")

findMeanAbsoluteError(y_true, y_predicted)
