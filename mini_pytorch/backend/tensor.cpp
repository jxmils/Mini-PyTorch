#include "tensor.h"
#include <iostream>  
#include <vector>
#include <memory>


// Constructor
Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad) {
    this->data = data;
    this->shape = shape;
    this->requires_grad = requires_grad;
    if (requires_grad) {
        this->grad_node = std::make_shared<Node>(0.0f);
    }
}

std::vector<int> Tensor::get_shape() const {
    return this->shape;
}

Tensor Tensor::operator+(const Tensor& other) {
    if (this->shape != other.shape) {
        throw std::invalid_argument("Shape mismatch: " + this->shape_as_string() + 
                                    " cannot be added to " + other.shape_as_string());
    }

    std::vector <float> result_data(this->data.size());
    for (size_t i = 0; i < this->data.size(); i++) {
        result_data[i] = this->data[i] + other.data[i];
    }

    Tensor result(result_data, this->shape, this->requires_grad || other.requires_grad);
    // If gradients are required, build a Node for this operation.
    if (result.requires_grad) {
        // Create a Node for the result and record the operation name ("add").
        result.grad_node = std::make_shared<Node>(result, "add");

        // Assume its grad_node is set.
        if (this->requires_grad) {
            result.grad_node->parents.push_back(this->grad_node);
        }
        if (other.requires_grad) {
            result.grad_node->parents.push_back(other.grad_node);
        }

        // Define the backward function.
        // For addition, d(result)/d(this) = 1 and d(result)/d(other) = 1.
        // When the backward pass runs, the gradient of the result is added to each parent's gradient.
        result.grad_node->backward_fn = [this, other, result]() mutable {
            // Propagate gradients to the left operand.
            if (this->requires_grad && this->grad_node) {
                for (size_t i = 0; i < this->grad_node->grad.data.size(); i++) {
                    this->grad_node->grad.data[i] += result.grad_node->grad.data[i];
                }
            }
            // Propagate gradients to the right operand.
            if (other.requires_grad && other.grad_node) {
                for (size_t i = 0; i < other.grad_node->grad.data.size(); i++) {
                    other.grad_node->grad.data[i] += result.grad_node->grad.data[i];
                }
            }
        };
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) {
    if (this->shape != other.shape) {
        throw std::invalid_argument("Shape mismatch: " + this->shape_as_string() + 
                                    " cannot be added to " + other.shape_as_string());
    }

    std::vector <float> result_data(this->data.size());
    for (size_t i = 0; i < this->data.size(); i++) {
        result_data[i] = this->data[i] - other.data[i];
    }

    return Tensor(result_data, this->shape); 
}

Tensor Tensor::operator*(const Tensor& other) {
    if (this->shape != other.shape) {
        throw std::invalid_argument("Shape mismatch: " + this->shape_as_string() + 
                                    " cannot be added to " + other.shape_as_string());
    }

    std::vector <float> result_data(this->data.size());
    for (size_t i = 0; i < this->data.size(); i++) {
        result_data[i] = this->data[i] * other.data[i];
    }

    return Tensor(result_data, this->shape); 
}

Tensor Tensor::operator/(const Tensor& other) {
    if (this->shape != other.shape) {
        throw std::invalid_argument("Shape mismatch: " + this->shape_as_string() + 
                                    " cannot be added to " + other.shape_as_string());
    }

    std::vector <float> result_data(this->data.size());
    for (size_t i = 0; i < this->data.size(); i++) {
        result_data[i] = this->data[i] / other.data[i];
    }

    return Tensor(result_data, this->shape); 
}

std::string Tensor::shape_as_string() const {
    std::string shape_str = "[";
    for (size_t i = 0; i < this->shape.size(); i++) {
        shape_str += std::to_string(this->shape[i]);
        if (i != this->shape.size() - 1) shape_str += ", ";
    }
    shape_str += "]";
    return shape_str;
}

void Tensor::print() const {
    std::cout << "Tensor(";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << data[i];
        if (i < data.size() - 1) std::cout << ", ";
    }
    std::cout << ", shape=" << shape_as_string() << ")" << std::endl;
}