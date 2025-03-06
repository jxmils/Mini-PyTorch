#ifndef NODE_H
#define NODE_H

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include "tensor.h"

// Node represents an operation in the computation graph.
// It stores the forward Tensor, its gradient, pointers to parent nodes,
// the backward function to propagate gradients, and the name of the operation.
class Node {
public:
    // The computed forward tensor.
    Tensor tensor;

    // The gradient accumulated during backpropagation.
    // Stored as a Tensor so that it matches the shape of the forward value.
    Tensor grad;

    // Pointers to parent nodes (the inputs used to compute this node).
    std::vector<std::shared_ptr<Node>> parents;

    // A backward function that computes and propagates gradients for this node's parents.
    std::function<void()> backward_fn;

    // The operator name (e.g., "add", "mul") for debugging and visualization.
    std::string op;

    // Constructor: initialize the node with a given tensor.
    // Also initializes the gradient tensor to zeros with the same shape.
    // Optionally, an operator name can be provided.
    Node(const Tensor& tensor, const std::string& op = "");

    // Destructor (empty because smart pointers handle memory automatically).
    ~Node();
};

#endif // NODE_H