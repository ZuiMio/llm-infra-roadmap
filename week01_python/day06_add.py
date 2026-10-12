"""
Week 01 - Additional Learning
Python scope, classes, iterators, and generator expressions.

Topics:
- local, nonlocal, and global scope
- class versus instance attributes
- method resolution order (MRO)
- name mangling
- dataclasses
- iter(), next(), and StopIteration
- custom iterators and generator functions
- generator expressions

Run this file directly to execute its examples.
"""

from dataclasses import dataclass


# ============================================================
# 1. Variable scope: local, nonlocal, and global
# ============================================================

spam = "initial global spam"


def scope_test():
    """Show where assignments made in nested functions take effect."""

    def do_local():
        spam = "local spam"
        print("Inside local function:", spam)

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


# ============================================================
# 2. Class attributes and bound methods
# ============================================================

class MyClass:
    """A simple class with an attribute and a method."""

    i = 12345

    def f(self):
        return "hello world"


def class_example():
    """Read a class attribute and invoke a bound method."""
    instance = MyClass()
    bound_method = instance.f

    print("Class attribute:", instance.i)
    print("Bound method:", bound_method())
    assert bound_method() == "hello world"


# ============================================================
# 3. Shared mutable class attributes: a deliberate pitfall
# ============================================================

class Dog:
    """Demonstrate the unintended sharing of a class-level list."""

    kind = "canine"
    tricks = []  # Intentionally shared by every Dog instance.

    def __init__(self, name):
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)


class IndependentDog:
    """Store mutable state on each instance instead."""

    kind = "canine"

    def __init__(self, name):
        self.name = name
        self.tricks = []

    def add_trick(self, trick):
        self.tricks.append(trick)


def dog_example():
    """Compare shared class state with independent instance state."""
    Dog.tricks.clear()  # Reset the intentional shared-state example.

    fido = Dog("Fido")
    buddy = Dog("Buddy")
    fido.add_trick("roll over")
    buddy.add_trick("play dead")

    print("Shared tricks (Fido):", fido.tricks)
    print("Shared tricks (Buddy):", buddy.tricks)
    print("Kinds:", fido.kind, buddy.kind)
    print("Names:", fido.name, buddy.name)

    assert fido.tricks is buddy.tricks

    alice = IndependentDog("Alice")
    bob = IndependentDog("Bob")
    alice.add_trick("sit")

    assert alice.tricks == ["sit"]
    assert bob.tricks == []
    assert alice.tricks is not bob.tricks


# ============================================================
# 4. Instance attributes can shadow class attributes
# ============================================================

class Warehouse:
    purpose = "storage"
    region = "west"


def warehouse_example():
    """Show that assignment to an instance does not change its class."""
    west = Warehouse()
    east = Warehouse()
    east.region = "east"

    print("West:", west.purpose, west.region)
    print("East:", east.purpose, east.region)
    print("Class defaults:", Warehouse.purpose, Warehouse.region)

    assert west.region == "west"
    assert east.region == "east"
    assert Warehouse.region == "west"


# ============================================================
# 5. Multiple inheritance and method resolution order
# ============================================================

class Base1:
    name = "Base1"


class Base2:
    name = "Base2"


class Base3:
    name = "Base3"


class DerivedClassName(Base1, Base2, Base3):
    pass


def inheritance_example():
    """Inspect the MRO that determines attribute lookup."""
    instance = DerivedClassName()
    print("Resolved name:", instance.name)
    print("MRO:", [cls.__name__ for cls in DerivedClassName.__mro__])
    assert instance.name == "Base1"


# ============================================================
# 6. Name mangling in subclasses
# ============================================================

class Mapping:
    """Keep a reference to the original update() implementation."""

    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)

    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)

    __update = update  # Mangled to _Mapping__update.


class MappingSubclass(Mapping):
    """Override update() without breaking the inherited constructor."""

    def update(self, keys, values):
        for item in zip(keys, values):
            self.items_list.append(item)


def name_mangling_example():
    """Show name mangling; it does not provide true access privacy."""
    base = Mapping([1, 2, 3])
    base.update([4, 5])
    print("Mapping:", base.items_list)

    child = MappingSubclass(["a", "b"])
    child.update(["x", "y"], [10, 20])
    print("MappingSubclass:", child.items_list)

    print("Original private alias exists:",
          "_Mapping__update" in Mapping.__dict__)

    assert base.items_list == [1, 2, 3, 4, 5]
    assert child.items_list == ["a", "b", ("x", 10), ("y", 20)]
    assert Mapping.__dict__["_Mapping__update"] is Mapping.__dict__["update"]
    assert "_Mapping__update" not in MappingSubclass.__dict__


# ============================================================
# 7. Dataclasses
# ============================================================

@dataclass
class Employee:
    name: str
    dept: str
    salary: int


def dataclass_example():
    """Create a class with a generated constructor and representation."""
    john = Employee("john", "computer lab", 1000)
    print("Employee:", john)
    print("Department:", john.dept)
    print("Salary:", john.salary)
    assert john.dept == "computer lab"
    assert john.salary == 1000


# ============================================================
# 8. iter(), next(), and iterator exhaustion
# ============================================================

def iterator_example():
    """Consume a string iterator and handle StopIteration safely."""
    iterator = iter("abc")

    print(next(iterator))  # a
    print(next(iterator))  # b
    print(next(iterator))  # c

    try:
        next(iterator)
    except StopIteration:
        print("The iterator is exhausted.")

    assert next(iterator, None) is None


# ============================================================
# 9. A custom one-shot reverse iterator
# ============================================================

class Reverse:
    """Iterate through a sequence in reverse order once."""

    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration

        self.index -= 1
        return self.data[self.index]


# ============================================================
# 10. A restartable iterable using a generator
# ============================================================

class Reverse2:
    """Generate a fresh reverse iterator on each iteration."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for index in range(len(self.data) - 1, -1, -1):
            yield self.data[index]


def reverse(data):
    """Yield the items of a sequence in reverse order."""
    for index in range(len(data) - 1, -1, -1):
        yield data[index]


def reverse_examples():
    """Compare a one-shot iterator with a restartable iterable."""
    one_shot = Reverse("spam")
    assert iter(one_shot) is one_shot
    print("Reverse:", list(one_shot))
    assert list(one_shot) == []

    restartable = Reverse2([1, 2, 3, 4])
    print("Reverse2:", list(restartable))
    assert list(restartable) == [4, 3, 2, 1]

    print("reverse():", list(reverse("golf")))
    assert "".join(reverse("golf")) == "flog"


# ============================================================
# 11. Generator expressions
# ============================================================

@dataclass
class Graduate:
    name: str
    gpa: float


def generator_expression_examples():
    """Use generator expressions in common data-processing tasks."""
    sum_of_squares = sum(i * i for i in range(10))
    print("Sum of squares:", sum_of_squares)
    assert sum_of_squares == 285

    xvec = [10, 20, 30]
    yvec = [7, 5, 3]
    dot_product = sum(x * y for x, y in zip(xvec, yvec))
    print("Dot product:", dot_product)
    assert dot_product == 260

    # Define the data before using it in generator expressions.
    page = ["Python is fun", "Python supports generators"]
    unique_words = {word for line in page for word in line.split()}
    print("Unique words:", sorted(unique_words))
    assert "Python" in unique_words

    graduates = [
        Graduate("Alice", 3.8),
        Graduate("Bob", 3.9),
        Graduate("Charlie", 3.7),
    ]
    valedictorian = max((student.gpa, student.name) for student in graduates)
    print("Highest GPA and name:", valedictorian)
    assert valedictorian == (3.9, "Bob")

    data = "golf"
    reversed_data = list(
        data[i] for i in range(len(data) - 1, -1, -1)
    )
    print("Reversed characters:", reversed_data)
    assert reversed_data == ["f", "l", "o", "g"]


# ============================================================
# Main
# ============================================================

def main():
    """Run and verify all examples."""
    print("=== Scope ===")
    scope_test()
    print("Global scope:", spam)
    assert spam == "global spam"

    print("\n=== Class and bound method ===")
    class_example()

    print("\n=== Shared class state ===")
    dog_example()

    print("\n=== Warehouse ===")
    warehouse_example()

    print("\n=== Multiple inheritance ===")
    inheritance_example()

    print("\n=== Name mangling ===")
    name_mangling_example()

    print("\n=== Dataclass ===")
    dataclass_example()

    print("\n=== Iterator exhaustion ===")
    iterator_example()

    print("\n=== Reverse iterators ===")
    reverse_examples()

    print("\n=== Generator expressions ===")
    generator_expression_examples()

    print("\nAll additional examples passed!")


if __name__ == "__main__":
    main()
