cmake_minimum_required(VERSION 3.14)
enable_testing()



project(MyProjectTest LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.11.0.zip # Or a specific tag/commit
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(test)
add_subdirectory(src)