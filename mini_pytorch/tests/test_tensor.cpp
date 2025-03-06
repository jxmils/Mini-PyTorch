#include "gtest/gtest.h"
#include "tensor.h"
#include <vector>
#include <stdexcept>

// Test that a tensor is created correctly.
TEST(TensorTest, Creation) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    std::vector<int> expected_shape = {4};
    EXPECT_EQ(t.get_shape(), expected_shape);
    
    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(t.data, expected_data);
}

// Test that the shape_as_string method returns the expected string.
TEST(TensorTest, ShapeAsString) {
    Tensor t({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_EQ(t.shape_as_string(), "[3]");
}

// Test tensor addition: element-wise addition.
TEST(TensorTest, Addition) {
    Tensor t1({1.0f, 2.0f, 3.0f}, {3});
    Tensor t2({4.0f, 5.0f, 6.0f}, {3});
    
    Tensor result = t1 + t2;
    std::vector<int> expected_shape = {3};
    EXPECT_EQ(result.get_shape(), expected_shape);
    
    std::vector<float> expected_data = {5.0f, 7.0f, 9.0f};
    EXPECT_EQ(result.data, expected_data);
}

// Test tensor subtraction.
TEST(TensorTest, Subtraction) {
    Tensor t1({1.0f, 2.0f, 3.0f}, {3});
    Tensor t2({4.0f, 5.0f, 6.0f}, {3});
    
    Tensor result = t1 - t2;
    std::vector<int> expected_shape = {3};
    EXPECT_EQ(result.get_shape(), expected_shape);
    
    std::vector<float> expected_data = {-3.0f, -3.0f, -3.0f};
    EXPECT_EQ(result.data, expected_data);
}

// Test tensor element-wise multiplication.
TEST(TensorTest, Multiplication) {
    Tensor t1({1.0f, 2.0f, 3.0f}, {3});
    Tensor t2({4.0f, 5.0f, 6.0f}, {3});
    
    Tensor result = t1 * t2;
    std::vector<int> expected_shape = {3};
    EXPECT_EQ(result.get_shape(), expected_shape);
    
    std::vector<float> expected_data = {4.0f, 10.0f, 18.0f};
    EXPECT_EQ(result.data, expected_data);
}

// Test tensor element-wise division.
TEST(TensorTest, Division) {
    Tensor t1({1.0f, 2.0f, 3.0f}, {3});
    Tensor t2({4.0f, 5.0f, 6.0f}, {3});
    
    Tensor result = t1 / t2; // Assumes operator/ is implemented.
    std::vector<int> expected_shape = {3};
    EXPECT_EQ(result.get_shape(), expected_shape);
    
    // Expected: 1/4, 2/5, 3/6
    std::vector<float> expected_data = {0.25f, 0.4f, 0.5f};
    ASSERT_EQ(result.data.size(), expected_data.size());
    for (size_t i = 0; i < expected_data.size(); i++) {
        EXPECT_NEAR(result.data[i], expected_data[i], 1e-5);
    }
}

// Test that a shape mismatch in an operation throws an error.
TEST(TensorTest, ShapeMismatchError) {
    Tensor t1({1.0f, 2.0f, 3.0f}, {3});
    Tensor t2({4.0f, 5.0f, 6.0f, 7.0f}, {4});
    EXPECT_THROW({
        Tensor result = t1 + t2;
    }, std::invalid_argument);
}

// reminder: main() is provided automatically when linking against gtest_main.