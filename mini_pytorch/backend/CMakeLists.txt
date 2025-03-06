set(BACKEND_SRC tensor.cpp)

add_library(backend ${BACKEND_SRC})

target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})