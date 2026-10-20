import numpy as np
import copy


a = np.array([[0,0], [0,0]])
print("a:\n", a)
b = a[:1]   # 通过切片等方式创建的视图, 如果你希望切片是独立的，互不影响，你需要显式地调用 .copy() 方法
b[0][0] = 1
print("a:\n", a)
print("b:\n", b)
print(b.base is a)
print(np.shares_memory(a, b))
del a, b
print("*"*60)

a = np.array([[0,0], [0,0]])
print("a:\n", a)
b = a[[0,1]]  # 花式索引 / 整数数组索引（Advanced Indexing），不共享内存
b[0][0] = 1
print("a:\n", a)
print("b:\n", b)
print(b.base is a)
print(np.shares_memory(a, b))
del a, b
print("*"*60)

##### array #####
a = np.array([[0,0], [0,0]])
print("a:\n", a)
b = a.copy()
b[0][0] = 1
print("a:\n", a)
print("b:\n", b)
print(b.base is a)
print(np.shares_memory(a, b))
del a, b
print("="*30)

a = np.array([[0,0], [0,0]], dtype=object)
print("a:\n", a)
b = a.copy()
b[0][0] = 1
print("a:\n", a)
print("b:\n", b)
print(b.base is a)
print(np.shares_memory(a, b))
del a, b
print("="*30)

a = np.array([[0,0], 1], dtype=object)
print("a:\n", a)
b = a.copy()
b[0][0] = 1
print("a:\n", a)
print("b:\n", b)
print(b.base is a)
print(np.shares_memory(a, b))
del a, b
print("="*30)

##### list #####
a = [[0,0], [0,0]]
print("a:\n", a)
b = a.copy()
b[0][0] = 1
print("a:\n", a)
print("b:\n", b)
del a, b
print("="*30)

a = [[0,0], 1]
print("a:\n", a)
b = a.copy()
b[0][0] = 1
print("a:\n", a)
print("b:\n", b)
del a, b
print("="*30)

a = np.array([[0,0], 1])    # illegal
print("a:\n", a)
b = a.copy()
b[0][0] = 1
print("a:\n", a)
print("b:\n", b)
del a, b
print("="*30)

