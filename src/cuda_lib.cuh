#pragma once
#include <stdio.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <complex>


// Function to convert compute capability to the number of cores per SM
int coresPerSM(int major, int minor) ;

void print_cuda_props() ;

std::vector<std::complex<float>> resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(const std::vector<int>& goldCode, float frequencyHz = 0);

float crossCorrelationCUDA(const std::vector<std::complex<float>>& signal1,
                           const std::vector<std::complex<float>>& signal2,
                           int lag);

std::complex<float> freq_shift_correlateCUDA(const std::vector<int>& goldCode, float freqShiftHz , const std::vector<std::complex<float>>& inputSignal, int lag);

std::complex<float> freq_shift_correlateLimitedSearchCUDA(const std::vector<int>& goldCode, float freqShiftHz ,
                                 const std::vector<std::complex<float>>& inputSignal, int lag, int bench = 0, int should_update = 0);

