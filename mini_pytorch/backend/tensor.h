#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>

#include "node.h"

class Tensor {
public:
    // The raw data stored in this tensor.
    std::vector<float> data;
    // The shape of the tensor.
    std::vector<int> shape;
    // Indicates if gradients should be tracked.
    bool requires_grad;
    // Pointer to the computation graph node for autograd.
    std::shared_ptr<Node> grad_node;

    // Constructor: initializes data, shape, and optionally requires_grad.
    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false);

    // Arithmetic operators.
    Tensor operator+(const Tensor& other);
    Tensor operator-(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor operator/(const Tensor& other);

    // Accessor methods.
    std::vector<int> get_shape() const;
    std::string shape_as_string() const;

    // Utility: print the tensor.
    void print() const;
};

#endif // TENSOR_H