#!/usr/bin/env python3  # Remove this if you want to run it on windows

import numpy as np

y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])

def findMeanAbsoluteError(y_true, y_predicted):
    epsilon = 1e-15
    total_error = 0
    # This is how you iterate through two arrays in one line.
    for i, j in zip(y_predicted, y_true):
        total_error = abs(i - j)
    print(f"Total Error: {total_error}")

    mean_absolute_error = total_error / len(y_true)

    print(f"Mean Absolute Error: {mean_absolute_error}")

findMeanAbsoluteError(y_true, y_predicted)
