# Deep Learning Library

A from-scratch deep learning framework in modern C++ with Python bindings.

## Motivation

Built to understand deep learning frameworks from first principles - from computational graphs to gradient computation to optimization algorithms.

## Running examples

For some examples on Python interface, see tests/python.

## Features

- **Computational Graph**: Dynamic graph construction with automatic differentiation
- **Core Components**: 
  - Automatic differentiation (autograd)
  - Backpropagation engine
  - Neural network layers
  - Training framework (optimizers, loss functions, layers, and networks)
- **Example code**: Full MNIST dataset training example
- **Python Interface**: Seamless integration via Boost.Python
- **Clean Architecture**: Modular design, maintainable and extensible
- **CI/CD**: Automated testing with GTest and GitHub Actions

## Tech Stack

- C++17/20
- CMake build system
- Boost.Python for Python bindings
- Python 3 for library interface and examples
- Google Test (GTest) and PyTest for unit testing
- GitHub Actions for CI/CD

## Current Status

🚧 **Work in Progress** - Implementing additional layers and optimizations

Roadmap:
- [x] Python Binding Unit Tests
- [x] Optimizers and training framework
- [x] MNIST example
- [ ] CUDA mode for operations
- [ ] Additional layer types (Conv2D, Dropout, etc.)
- [ ] AlexNet reference implementation
- [ ] Docker deployment example

## Building
```bash
mkdir build && cd build
cmake ..
make
```

### Building with CUDA

To build with CUDA, a flag has to be supplied to CMake.

```bash
cmake --DCUDA=On ..
```


## Running Unit Tests

Compile with building tests enabled: 

```bash
mkdir build && cd build
cmake -DBUILD_TESTS=On ..
make
ctest .
```

## Required

- Compiler capable of C++23 at least (we test with gcc 13.3.0)
- Boost Python
- Cmake > 3.28
- Python 3 (we test with 3.10, but it should work with any version)
- numpy 1.26.4
- pytest and GTest for unit tests (we use pytest=9.0.2)
- Google Benchmark for benchmarking

## Troubleshooting

### Building on Windows

The implementation of the Python wrapper does not work on MSVC6/7 in its current form. This is due to an issue that arises from Boost Python in combination with these compilers. Workarounds are proposed, but not implemented. More information here [here](https://beta.boost.org/doc/libs/develop/libs/python/doc/html/tutorial/tutorial/exposing.html).

## License

MIT
