#include "node.h"

// Constructor: copies the forward tensor, initializes grad to a zero tensor,
// and sets the operator name for debugging.
Node::Node(const Tensor& tensor, const std::string& op)
    : tensor(tensor),
      grad(std::vector<float>(tensor.data.size(), 0.0f), tensor.shape, false),
      op(op)
{
    // The parents vector is automatically initialized as empty.
    // backward_fn can be set later when an operation is performed.
}

Node::~Node() {
    // No explicit cleanup needed; smart pointers handle memory.
}