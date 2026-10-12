

def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)


class MyClass:
    """A simple example class"""
    i = 12345

    def f(self):
        return 'hello world'
x = MyClass()
xf = x.f
print(xf())


class Dog:

    kind = 'canine'         # 全インスタンスで共有されるクラス変数
    tricks = []             # 类变量的错误用法

    def __init__(self, name):
        self.name = name    # インスタンスごとに固有のインスタンス変数

    def add_trick(self, trick):
        self.tricks.append(trick)

d = Dog('Fido')
e = Dog('Buddy')
d.add_trick('roll over')
e.add_trick('play dead')
print(d.tricks)                # 非预期地被所有的 Dog 实例所共享
print(d.kind)                  # すべての犬で共有
print(e.kind)                  # すべての犬で共有
print(d.name)                  # d 固有
print(e.name)                  # e 固有

class Warehouse:
   purpose = 'storage'
   region = 'west'
w1 = Warehouse()
w2 = Warehouse()
w2.region = 'east'
print(w1.purpose, w1.region)
print(w2.purpose, w2.region)
print(Warehouse.purpose, Warehouse.region)


class Base1:
    name = 'Base1'
class Base2:
    name = 'Base2'
class Base3:
    name = 'Base3'

class DerivedClassName(Base1, Base2, Base3):
    pass

x = DerivedClassName()
print(x.name)  # 输出 'Base1'，因为 Base1 是第一个基类，按照 MRO（方法解析顺序）规则，优先使用 Base1 的 name


class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)

    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)

    __update = update   # 原始 update() 的私有别名


class MappingSubclass(Mapping):

    def update(self, keys, values):
        # 引入新的 update() 签名
        # 但不会破坏继承来的 __init__()
        for item in zip(keys, values):
            self.items_list.append(item)


print("=== Mapping ===")
m = Mapping([1, 2, 3])
print(m.items_list)          # [1, 2, 3]

m.update([4, 5])
print(m.items_list)          # [1, 2, 3, 4, 5]


print("=== MappingSubclass ===")
ms = MappingSubclass(['a', 'b'])
print(ms.items_list)         # ['a', 'b']

ms.update(['x', 'y'], [10, 20])
print(ms.items_list)         # ['a', 'b', ('x', 10), ('y', 20)]


print("=== 名称修饰验证 ===")
print('_Mapping__update' in Mapping.__dict__)          # True
print('_Mapping__update' in MappingSubclass.__dict__)  # False
print(Mapping.__dict__['_Mapping__update'] is Mapping.__dict__['update'])  # True
# exec() や eval() へ渡されたコードでは、呼出し元のクラス名を現在のクラスと見なさないことに注意してください。
# 同じ制約が getattr() と setattr() と delattr() にも適用されます。また、__dict__ を直接参照するときにも適用されます。


from dataclasses import dataclass

@dataclass
class Employee:
    name: str
    dept: str
    salary: int

john = Employee('john', 'computer lab', 1000)
john.dept
john.salary

s = 'abc'
it = iter(s)
it

next(it)

next(it)

next(it)

next(it)


class Reverse:
    """对一个序列执行反向循环的迭代器。"""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]
rev = Reverse('spam')
iter(rev)

for char in rev:
    print(char)


class Reverse2:
    """可迭代对象：反向遍历序列，但不自己实现 __next__。"""
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # 生成器函数：自动拥有 __iter__ 和 __next__
        for index in range(len(self.data) - 1, -1, -1):
            yield self.data[index]

# 使用
r = Reverse2([1, 2, 3, 4])
for item in r:
    print(item)


def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]
for char in reverse('golf'):
    print(char)

sum(i*i for i in range(10))                 # sum of squares


xvec = [10, 20, 30]
yvec = [7, 5, 3]
sum(x*y for x,y in zip(xvec, yvec))         # dot product


unique_words = set(word for line in page  for word in line.split())

valedictorian = max((student.gpa, student.name) for student in graduates)

data = 'golf'
list(data[i] for i in range(len(data)-1, -1, -1))

