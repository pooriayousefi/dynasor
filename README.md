# Dynasor - Dynamic Tensor Library

[![Build Status](https://github.com/pooriayousefi/dynasor/actions/workflows/ci.yml/badge.svg)](https://github.com/pooriayousefi/dynasor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-green.svg)](https://cmake.org/)

A modern C++20 header-only library for implementing dynamically sized conceptual tensor data structures. Dynasor provides efficient, type-safe multi-dimensional arrays with support for parallel execution policies and modern C++ concepts.

## 🚀 Features

- **Header-Only**: Single header include for easy integration
- **C++20 Concepts**: Type-safe interfaces with concept constraints
- **Parallel Execution**: Support for STL execution policies (sequential, parallel)
- **Dynamic Dimensions**: Runtime-configurable tensor dimensions
- **Type Safety**: Template constraints for arithmetic types
- **Memory Efficient**: Contiguous memory layout for optimal cache performance
- **Cross-Platform**: Works on Linux (g++), macOS (clang++), and Windows (MSVC)

## 🎯 Quick Start

```cpp
#include "dynasor.h"
#include <initializer_list>
#include <execution>

int main() {
    // Create a 2x3x4 tensor with sequential execution
    std::initializer_list<size_t> dimensions{2, 3, 4};
    dynasor<float> tensor(std::execution::seq, 
                         dimensions.begin(), 
                         dimensions.end());
    
    // Create tensor with parallel uniform random initialization
    std::initializer_list<size_t> dims{3, 3};
    auto random_tensor = dynasor<int>::uniform_random(
        std::execution::par,
        dims.begin(), dims.end(),
        42,    // seed
        -10,   // min value
        10     // max value
    );
    
    return 0;
}
```

## 📚 API Reference

### Core Constructor

```cpp
template<execution_policy ExPo, integral_value_iterator DimIter>
dynasor(ExPo execution_policy, DimIter dim_begin, DimIter dim_end, T init_value = T{})
```

Creates a tensor with specified dimensions and optional initialization value.

### Random Initialization

```cpp
template<execution_policy ExPo, integral_value_iterator DimIter>
static dynasor uniform_random(ExPo execution_policy, 
                             DimIter dim_begin, DimIter dim_end,
                             unsigned seed, T min_val, T max_val)
```

Creates a tensor filled with uniformly distributed random values.

### Data Initialization

```cpp
template<execution_policy ExPo, integral_value_iterator DimIter, arithmetic_value_iterator ValIter>
dynasor(ExPo execution_policy, 
        DimIter dim_begin, DimIter dim_end,
        ValIter val_begin, ValIter val_end)
```

Creates a tensor from existing data with specified dimensions.

## 🔧 Concepts and Type Safety

Dynasor uses C++20 concepts for compile-time type validation:

- `arithmetic`: Accepts integral or floating-point types
- `integral_value_iterator`: Iterators over integral types
- `floating_point_value_iterator`: Iterators over floating-point types  
- `arithmetic_value_iterator`: Iterators over arithmetic types
- `execution_policy`: STL execution policy concepts

## 🚀 Performance Features

- **Parallel Execution**: Leverage `std::execution::par` for multi-core operations
- **Contiguous Memory**: Single vector storage for optimal cache locality
- **SIMD-Friendly**: Memory layout optimized for vectorization
- **Compile-Time Optimization**: Template specialization and concept constraints

## 🏗️ Building from Source

```bash
# Clone repository
git clone https://github.com/pooriayousefi/dynasor.git
cd dynasor

# Build with CMake
cmake --preset=default
cmake --build build/default

# Run example
./build/default/dynasor_example
```

## 📊 Use Cases

- **Machine Learning**: Neural network weight tensors and activations
- **Scientific Computing**: Multi-dimensional data analysis
- **Image Processing**: Multi-channel image representations
- **Numerical Simulations**: Grid-based computations
- **Computer Graphics**: Volumetric data and textures

## 🔧 Requirements

- C++20 compatible compiler (GCC 11+, Clang 13+, MSVC 2022+)
- CMake 3.20 or later
- Standard library with execution policy support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

---

**Author**: [Pooria Yousefi](https://github.com/pooriayousefi)  
**Repository**: [https://github.com/pooriayousefi/dynasor](https://github.com/pooriayousefi/dynasor)
