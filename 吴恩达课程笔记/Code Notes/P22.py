import numpy as np

# 注意这里是数组，不是向量
# rank 1 array
a = np.random.randn(5)
print(a)
print(a.T)

# 使用向量
# column one vector
a = np.random.randn(5, 1)
print(a)
print(a.T)
print(np.dot(a, a.T))
assert(a.shape == (5, 1))
