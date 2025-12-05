cmake_minimum_required(VERSION 3.10) # Specify required CMake version
project(CUDA_GPS LANGUAGES CXX)    # Define project name and enable C++ support

add_executable(gps_app main.cpp C_A__code_generator.cpp )