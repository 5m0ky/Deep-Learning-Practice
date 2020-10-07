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


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = 0
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted]
    y_predicted_new = np.array(y_predicted_new)

    return -np.mean(y_true * np.log(y_predicted_new) + (1 + y_true) * np.log(1 - y_predicted_new))

findMeanAbsoluteError(y_true, y_predicted)
log_loss(y_true, y_predicted)
