# Mini-PyTorch

Mini-PyTorch is a personal deep learning framework inspired by PyTorch. This project is a hands-on exploration into the inner workings of deep learning frameworks, focusing on the implementation of tensors, automatic differentiation (autograd), and neural network building blocks in C++ with optional Python integration.

## Features

- **Tensor Operations:**  
  Implement basic tensor arithmetic (addition, subtraction, multiplication, division) with shape checking and error handling.

- **Autograd Engine:**  
  Build a custom autograd system that dynamically constructs a computation graph using a `Node` class. Each node stores the forward tensor, its gradient, parent nodes, and a backward function for gradient propagation.

- **Debugging & Visualization:**  
  Each node records its operator type (e.g., "add", "mul") for easier debugging and later visualization of the computation graph.

- **Extensible Design:**  
  The project is designed to be modular and extensible, allowing you to gradually add more features like matrix multiplication, neural network layers, optimizers, and eventually even GPU support.

## Getting Started

### Prerequisites

- **C++ Compiler:** A modern C++ compiler supporting C++17 (GCC, Clang, etc.).
- **CMake:** Version 3.10 or higher.
- **GoogleTest:** The project uses GoogleTest for unit testing, which is fetched automatically via CMake's FetchContent.
- **Optional:** Python (and necessary libraries) if you plan on building Python bindings or running the example scripts.

### Building the Project

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Mini-PyTorch.git
   cd Mini-PyTorch/mini_pytorch
   ```
   
2. **Create a Build Directory and Configure with CMake:**
    ```bash
    rm -rf build
    mkdir build && cd build
    cmake ..
    ```

3. **Compile the Project:**
   ```bash
   make
   ```

### Running Tests
From the build directory run:
  ```bash
  ctest --output-on-failure
  ```
   
