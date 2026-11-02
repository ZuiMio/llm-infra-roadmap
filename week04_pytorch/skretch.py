



import torch
import torch.nn.functional as F

vocab_size = 4
embedding_size = 3

# Embedding
emb = torch.nn.Embedding(vocab_size, embedding_size)

# 比較用Linear（Biasなし）
linear = torch.nn.Linear(
    vocab_size,
    embedding_size,
    bias=False
)

# 同じ重みを設定する
with torch.no_grad():
    linear.weight.copy_(emb.weight.T)

# 文字ID
ids = torch.tensor([1, 0, 2])

# 方法1：Embedding
output_emb = emb(ids)

# 方法2：One-Hot + Linear
one_hot = F.one_hot(
    ids,
    num_classes=vocab_size
).float()

output_linear = linear(one_hot)

print(output_emb)
print(output_linear)

print(torch.allclose(output_emb, output_linear))
# True


# import torch

# x = torch.tensor(3.0, requires_grad=True)
# y = x ** 2

# print(x)
# print(y)
# print(x.grad)
# print(y.grad_fn)

# y.backward()

# print(x.grad)




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