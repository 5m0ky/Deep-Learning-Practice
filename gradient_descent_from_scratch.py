#---------------------------------------------------------------------------------------------------------------------------------------
# Written by: 5m0ky
# Learnt from: Codebasics
# Codebasics github: https://github.com/codebasics
# Codebasics youtube: https://www.youtube.com/channel/UCh9nVJoWXmFb7sLApWGcLPQ
# This is just a snippet code for gradient descent i practiced while learning deep learning from codebasic's channel.
# My jupyter notebook was messed up so i thought to practice it all over again by writting it in order by organizing it.
#---------------------------------------------------------------------------------------------------------------------------------------

import numpy as np 
import math
import pandas as pd
from sklearn.model_selection import train_test_split

# Reading in the csv file
data = pd.read_csv("insurance.csv")

X_train, X_test, y_train, y_test = train_test_split(data[['age','affordibility']], data.bought_insurance, test_size = 0.2)

# Scaling the data
X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100

# Log loss formula implementation
def log_loss(y_true, y_predicted):
	epsilon = 1e-15 
	
	y_predicted_new = [max(i,1-epsilon) for i in y_predicted]
	y_predicted_new = [min(i,epsilon) for i in y_predicted_new]
	y_predicted_new = np.array(y_predicted_new)

	return -np.mean(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))

# Sigmoid formula implementation
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Gradient descent formula implementation
def gradient_descent(age, affordibility, y_true, epochs):
	w1 = w2 = 1
	bias = 0
	rate = 0.5
	n = len(age)

	for i in range(epochs):
		weighted_sum  = w1 * age + w2 * affordibility + bias
		y_predicted = sigmoid(weighted_sum)
		loss = log_loss(y_true, y_predicted)

		# Derivatives
		w1d = (1/n) * np.dot(np.transpose(age),(y_predicted - y_true))
		w2d = (1/n) * np.dot(np.transpose(affordibility),(y_predicted - y_true))
		bias_d = np.mean(y_predicted - y_true)

		w1 = w1 - rate * w1d
		w2 = w2 - rate * w2d
		bias = bias - rate * bias_d

		print(f"Weight1: {w1}, Weight2: {w2}, Loss: {loss}, bias: {bias}, Epoch: {i}\n")

	return w1, w2, bias

gradient_descent(X_train_scaled['age'], X_train_scaled['affordibility'], y_train, 1000)