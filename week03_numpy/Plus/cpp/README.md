# C++ Side Track — Vector-Based Linear Forward

The Week 03 plan adds a small C++ side track alongside NumPy. This short exercise practices function signatures, loops, `std::vector<float>`, const references, mutable references, and explicit shape validation.

## Compile and run

From the repository root:

~~~bash
g++ -std=c++17 -Wall -Wextra -pedantic week03_numpy/Plus/cpp/linear_forward.cpp -o /tmp/week3_linear_forward
/tmp/week3_linear_forward
~~~

Expected output:

~~~text
4.5 4.5
~~~

## What to inspect

- `const std::vector<float>& input`: pass an existing vector by reference without copying it, and do not allow the function to change it.
- `std::vector<float>& values`: mutable reference; the caller's vector can be changed by the function.
- `std::vector<std::vector<float>>`: a simple representation of a matrix.
- `std::invalid_argument`: report incompatible dimensions instead of computing invalid results.
- Nested loops: each output feature is a sum over input features, then a bias is added.

This is the beginning of the C++ side track, not a replacement for later topics such as classes, templates, RAII, smart pointers, and move semantics.
