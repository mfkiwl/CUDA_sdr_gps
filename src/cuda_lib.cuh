#pragma once
#include <stdio.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <complex>


// Function to convert compute capability to the number of cores per SM
int coresPerSM(int major, int minor) ;

void print_cuda_props() ;

std::vector<std::complex<float>> resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(const std::vector<int>& goldCode, float frequencyHz = 0);