

add_executable(my_test_runner gold_code_test.cpp ../src/C_A__code_generator.cpp ../src/cuda_lib.cu)
target_include_directories(my_test_runner
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(my_test_runner PRIVATE GTest::gtest GTest::gtest_main)