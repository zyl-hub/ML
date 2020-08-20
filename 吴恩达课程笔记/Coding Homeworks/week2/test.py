import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()

# index = 25
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("train_set_y=" + str(train_set_y)) #你也可以看一下训练集里面的标签是什么样的。

# 打印出当前的训练标签值
# 使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
# print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
# 只有压缩后的值才能进行解码操作
# print("y=" + str(train_set_y[:, index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture")

m_train = train_set_y.shape[1]  # 训练集里图片的数量。
m_test = test_set_y.shape[1]  # 测试集里图片的数量。
num_px = train_set_x_orig.shape[1]  # 训练、测试集里面的图片的宽度和高度（均为64x64）。

# 现在看一看我们加载的东西的具体情况
print ("训练集的数量: m_train = " + str(m_train))
print ("测试集的数量 : m_test = " + str(m_test))
print ("每张图片的宽/高 : num_px = " + str(num_px))
print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
print ("测试集_标签的维数: " + str(test_set_y.shape))

# X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
# 将训练集的维度降低并转置。
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print ("测试集_标签的维数 : " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s


def init(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b


def forward_and_backward(w, b, X, Y):
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    m = X.shape[1]
    J = -np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

    dZ = A - Y
    dw = np.dot(X, dZ.T)/m
    db = np.sum(dZ)/m

    J = np.squeeze(J)

    grads = {
        "dw": dw,
        "db": db
    }
    return (grads, J)


def optimize(w, b, X, Y, num_iterations, alpha):
    J = []

    for i in range(num_iterations):

        grads, J = forward_and_backward(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - alpha * dw
        b = b - alpha * db

    params = {
        "w" : w,
        "b" : b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return (params, grads, J)


def predict_var(w, b, X):
    # m是训练集和测试集的labeled图片数目
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 20000, alpha = 0.5):

    # shape[0]读取第一维度长度
    w, b = init(X_train.shape[0])

    parameters, grads, J = optimize(w, b, X_train, Y_train, num_iterations, alpha)

    w, b = parameters["w"], parameters["b"]

    Y_prediction_test = predict_var(w, b, X_test)
    Y_prediction_train = predict_var(w, b, X_train)

    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs": J,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": alpha,
        "num_iterations": num_iterations}
    return d

print("====================测试model====================")
#这里加载的是真实的数据，请参见上面的代码部分。
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, alpha = 0.005)