import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

iterations = 2000
train_rate = 0.005



train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

imgAmount = train_set_x.shape[1]
imgAmountOfTest = test_set_x.shape[1]

w = np.zeros(shape=(test_set_x.shape[0], 1))
b = 0
Z = np.dot(w.T, train_set_x) + b
A = 1/(1 + np.exp(-Z))

Loss = -np.sum((train_set_y * np.log(A) + (1 - train_set_y) * np.log(1 - A))) / imgAmount

dZ = A - train_set_y
dw = np.dot(train_set_x, dZ.T)/imgAmount
db = np.sum(dZ)/imgAmount

for i in range(iterations):
    Z = np.dot(w.T, train_set_x) + b
    A = 1 / (1 + np.exp(-Z))

    Loss = -np.sum((train_set_y * np.log(A) + (1 - train_set_y) * np.log(1 - A))) / imgAmount

    dZ = A - train_set_y
    dw = np.dot(train_set_x, dZ.T) / imgAmount
    db = np.sum(dZ) / imgAmount

    w = w - train_rate * dw
    b = b - train_rate * db

Y_prediction_test = np.zeros((1, imgAmountOfTest))
w = w.reshape(test_set_x.shape[0], 1)
A = 1/(1 +np.exp(-(np.dot(w.T, test_set_x) + b)))
for i in range(A.shape[1]):
    Y_prediction_test[0, i] = 1 if A[0, i] > 0.5 else 0

print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100), "%")
