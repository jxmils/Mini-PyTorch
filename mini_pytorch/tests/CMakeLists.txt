cmake_minimum_required(VERSION 3.10)
project(MiniPyTorchTests)

set(CMAKE_CXX_STANDARD 17)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${PROJECT_SOURCE_DIR}/../backend)

add_executable(tensor_tests test_tensor.cpp ../backend/tensor.cpp)
target_link_libraries(tensor_tests gtest_main)

include(GoogleTest)
gtest_discover_tests(tensor_tests)