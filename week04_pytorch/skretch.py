


# import numpy as np

# # 创建一个 0 行 5 列的空数组
# empty_matrix = np.zeros((0, 5))

# print(empty_matrix.shape)  # 输出: (0, 5)
# print(empty_matrix.size)   # 输出: 0 (总元素个数为 0)





# import numpy as np

# x = np.array([
#     [1.0, 10.0],
#     [3.0, 14.0],
# ])

# mean = np.mean(x, axis=0, keepdims=True)
# centered = x - mean
# print(centered)
# mean = np.mean(x, axis=0)
# centerd = x - mean
# print(centered)







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








# import numpy as np

# a = np.asarray(
#     [
#         [1, 2],
#         [3, 4]
#     ]
# )

# b = np.asarray(
#     [1, 2]
# )
# c = b[:, None]

# print(a * b)
# print(a * c)