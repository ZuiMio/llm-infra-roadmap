
# import numpy as np

# # 假设原数组 x 的形状是 (3, 4, 5)
# x = np.zeros((3, 4, 5))
# # print(x)

# y = np.asarray(x)[:, np.newaxis, :, :]  # new dimension was inserted
# print(y)
# # 1. 使用 np.newaxis
# print(x[:, np.newaxis, :, :].shape)  # 输出: (3, 1, 4, 5)

# # 2. 使用 None (效果完全相同)
# print(x[:, None, :, :].shape)         # 输出: (3, 1, 4, 5)

import numpy as np

a = np.asarray(
    [
        [1, 2],
        [3, 4]
    ]
)

b = np.asarray(
    [1, 2]
)
c = b[:, None]

print(a * b)
print(a * c)