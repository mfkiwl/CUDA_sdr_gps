// src/cuda_lib.cu
#include "cuda_lib.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <string>

// For time measurment
#include <cstdint>
#include <cuda/std/chrono>
#include <iostream>


// const int SAMPLES_PER_SECOND = 10230000;//4092000; //SAMPLES_PER_MS00;
// // const int SAMPLES_PER_SECOND =    4092000; //SAMPLES_PER_MS00;

// const int SAMPLES_PER_MS = SAMPLES_PER_SECOND / 1000;
// const int BLOCK_SIZE = (SAMPLES_PER_MS/256)+1;
// const int SAMPLES_PER_CHIP = 10;
// const int SAMPLES_PER_CHIP_FRAC = 1;
// const int  OUTPUT_SIZE = 2*BLOCK_SIZE; //2*10230;//2*(BLOCK_SIZE/256);



const int SAMPLES_PER_SECOND = 25000000;//4092000; //SAMPLES_PER_MS00;

const int SAMPLES_PER_MS = SAMPLES_PER_SECOND / 1000;
const int BLOCK_SIZE = (SAMPLES_PER_MS/256)+1;
const int SAMPLES_PER_CHIP = 25;
const float SAMPLES_PER_CHIP_FRAC = 1.023;
const int  OUTPUT_SIZE = 2*BLOCK_SIZE;




// __global__ void gpu_freq_shift_correlate(float* cuda_signalI1, float* cuda_signalQ1, float* cuda_signalI2, float* cuda_signalQ2, float freqShiftHz,int lag)
__global__ void  gpu_freq_shift_correlate(float phase, float* cuda_signalI1, float* cuda_signalQ1, int* cuda_goldCode, float freqShiftHz,int lag, float *cuda_output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_dataI[256];
    __shared__ float shared_dataQ[256];

    //0..20/1.023
    float gold_code = cuda_goldCode[((  (int) ((i+ lag)*SAMPLES_PER_CHIP_FRAC))  /SAMPLES_PER_CHIP)%1023] == 1 ? 1.0f : -1.0f;

    //  (A * Cos + jA * SIN) * B(cos + j sin) = (A*B * cos - A*B * sin) + j(A*B * sin + A*B * cos)
    float r1 =  gold_code * cosf(phase + 2.0f * M_PI * freqShiftHz * ((float)i/SAMPLES_PER_SECOND));
    float i1 = -1*gold_code * sinf(phase + 2.0f * M_PI * freqShiftHz * ((float)i/SAMPLES_PER_SECOND));

    float real1 = cuda_signalI1[i] * r1 - cuda_signalQ1[i] * i1;
    // imag1 = cuda_signalQ1[i] * real1 + cuda_signalI1[i] * imag1;
    // imag1 = gold_code * cosf(2.0f * M_PI * freqShiftHz * ((float)i/SAMPLES_PER_SECOND)) * imag1 ;// - gold_code * sinf(-2.0f * M_PI * freqShiftHz * ((float)i/SAMPLES_PER_SECOND)) * real1;
    // image1 = cos(a) * sin(b) - sin(a) * cos(b) ;
    // float imag1 = r1 * cuda_signalQ1[i] - sinf(2.0f * M_PI * freqShiftHz * ((float)i/SAMPLES_PER_SECOND)) * cosf(2.0f * M_PI * freqShiftHz * ((float)i/SAMPLES_PER_SECOND)) ;
    float imag1 = r1 * cuda_signalQ1[i] +i1 * cuda_signalI1[i] ;

    shared_dataI[threadIdx.x] = real1;
    shared_dataQ[threadIdx.x] = imag1;

    // __syncthreads();


    if (threadIdx.x == 0)
    {
    // Reduction within the block
        for (int i = 0 ;  i < blockDim.x ; i++)
        {
            shared_dataI[0] += shared_dataI[i];
            shared_dataQ[0] += shared_dataQ[i];
        }
    }

    // // // for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    // // //     if (threadIdx.x < stride) {
    // // //         shared_dataI[threadIdx.x] += shared_dataI[threadIdx.x + stride];
    // // //         shared_dataQ[threadIdx.x] += shared_dataQ[threadIdx.x + stride];
    // // //     }
    // // //     __syncthreads();
    // // // }

    if (threadIdx.x == 0)
    {
        // cuda_output[blockIdx.x*2] = shared_dataI[0] ;
        cuda_output[blockIdx.x*2] = shared_dataI[0];
        cuda_output[blockIdx.x*2+1] = shared_dataQ[0];

    }

    // cuda_output[2*i] = real1;
    // cuda_output[2*i+1] = imag1;


}


std::complex<float> freq_shift_correlateLimitedSearchCUDA(const std::vector<int>& goldCode, float freqShiftHz , const std::vector<std::complex<float>>& inputSignal, int limitedLag, int bench)
{
    int arrGoldCode[1023];
    float output[OUTPUT_SIZE];

    static int prev_I_sign = 0;
    static int prev_Q_sign = 0;

    int *cuda_goldCode;

    for (size_t i = 0; i < 1023; ++i) {
        arrGoldCode[i] = (goldCode[i] == 1? 1.0f : -1.0f);
    }


    cudaMalloc(&cuda_goldCode, 1023 * sizeof(float));
    cudaMemcpy(cuda_goldCode, arrGoldCode, 1023 * sizeof(int), cudaMemcpyHostToDevice);


    float signalI1[SAMPLES_PER_MS];
    float signalQ1[SAMPLES_PER_MS];

    for (size_t i = 0; i < SAMPLES_PER_MS; ++i) {
        signalI1[i] = inputSignal[i].real();
        signalQ1[i] = inputSignal[i].imag();
    }



    float *cuda_signalI1;
    float *cuda_signalQ1;
    float *cuda_output;

    cudaMalloc(&cuda_signalI1, SAMPLES_PER_MS * sizeof(float));
    cudaMalloc(&cuda_signalI1, SAMPLES_PER_MS * sizeof(float));
    cudaMalloc(&cuda_signalQ1, SAMPLES_PER_MS * sizeof(float));
    cudaMalloc(&cuda_output, OUTPUT_SIZE * sizeof(float));

    cudaMemset(cuda_signalI1, 0, SAMPLES_PER_MS * sizeof(float));
    cudaMemset(cuda_signalQ1, 0, SAMPLES_PER_MS * sizeof(float));


    cudaMemcpy(cuda_signalI1, signalI1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_signalQ1, signalQ1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);

    float max_cross = 0 ;
    float max_freq = 0 ;
    int max_lag = 0 ;
    std::complex<float> max_sum = std::complex<float>(0.0f, 0.0f);
    for (int lag = (limitedLag - 30 )%SAMPLES_PER_MS; (lag < limitedLag + 30)% SAMPLES_PER_MS; lag += 3)
    // for (lag = 0; lag < SAMPLES_PER_MS; lag += 3)
    // lag = 1230;
    {

        // for (freqShiftHz = -5000; freqShiftHz <= 5000; freqShiftHz += 250)
        // freqShiftHz = -3500;//3250;
        {
            float phase = 2 * M_PI * 1/1000 * freqShiftHz * bench;

            gpu_freq_shift_correlate<<<(SAMPLES_PER_MS/256)+1, 256>>>(phase,cuda_signalI1, cuda_signalQ1, cuda_goldCode,  freqShiftHz, lag, cuda_output);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
                exit(1); // If CUDA fails, there is nothing we can do
            }


            cudaMemcpy(output, cuda_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            // for (int i = 0; i <  OUTPUT_SIZE/2; i++) {
            //     printf("(%f,%f)", output[2*i], output[2*i+1]);
            // }

            // exit(1);

            std::complex<float> sum = std::complex<float>(0.0f, 0.0f);
            for (size_t i = 0; i < BLOCK_SIZE; i++) {
                sum += std::complex<float>(output[2*i], output[2*i+1]);
            }
            float realSum = std::abs(sum);

            if (realSum > max_cross) {
                max_cross = realSum;
                max_freq = freqShiftHz;
                max_lag = lag;
                max_sum = sum;
                // printf( "Sat #%d freqShiftHz:%f Lag:%d Cross:%d\n", i, freqShiftHz, lag, cross_cuda);
            }
            // printf( "Lag: %d  FreqShiftHz:%f Cross:(%f,%f)\n", lag, freqShiftHz, sum.imag(), sum.real());
            // printf("%f,", realSum);
        }
        // printf("\n");/
    }

    int current_I_sign = (max_sum.real() >= 0) ? 1 : -1;
    int current_Q_sign = (max_sum.imag() >= 0) ? 1 : -1;
    bool sign_changed = (prev_I_sign != current_I_sign) && (prev_Q_sign != current_Q_sign );
    prev_I_sign = current_I_sign;
    prev_Q_sign = current_Q_sign;

    printf("max cross:%f max freq:%f max lag:%d chips:%d sign_changed:%d\n", max_cross, max_freq, max_lag, max_lag/SAMPLES_PER_CHIP, sign_changed );


    cudaFree(cuda_output);
    cudaFree(cuda_signalI1);
    cudaFree(cuda_signalQ1);
    cudaFree(cuda_goldCode);

    return max_sum;// (int)realSum;


}


// Convert gold codes to baseband (complex-valued signal)
std::complex<float> freq_shift_correlateCUDA(const std::vector<int>& goldCode, float freqShiftHz , const std::vector<std::complex<float>>& inputSignal, int lag) {


    int arrGoldCode[1023];
    float output[OUTPUT_SIZE];

    int *cuda_goldCode;

    for (size_t i = 0; i < 1023; ++i) {
        arrGoldCode[i] = (goldCode[i] == 1? 1.0f : -1.0f);
    }


    cudaMalloc(&cuda_goldCode, 1023 * sizeof(float));
    cudaMemcpy(cuda_goldCode, arrGoldCode, 1023 * sizeof(int), cudaMemcpyHostToDevice);


    // for (int i = 0; i < SAMPLES_PER_MS; i++)
    //     printf("%d\n", (int)((float)i*SAMPLES_PER_CHIP_FRAC));
    //     // printf("%d\n", (((i+ lag)*SAMPLES_PER_CHIP_FRAC)/SAMPLES_PER_CHIP)%1023);
    // exit(1);

    float signalI1[SAMPLES_PER_MS];
    float signalQ1[SAMPLES_PER_MS];

    for (size_t i = 0; i < SAMPLES_PER_MS; ++i) {
        signalI1[i] = inputSignal[i].real();
        signalQ1[i] = inputSignal[i].imag();
    }



    float *cuda_signalI1;
    float *cuda_signalQ1;
    float *cuda_output;

    cudaMalloc(&cuda_signalI1, SAMPLES_PER_MS * sizeof(float));
    cudaMalloc(&cuda_signalI1, SAMPLES_PER_MS * sizeof(float));
    cudaMalloc(&cuda_signalQ1, SAMPLES_PER_MS * sizeof(float));
    cudaMalloc(&cuda_output, OUTPUT_SIZE * sizeof(float));

    cudaMemset(cuda_signalI1, 0, SAMPLES_PER_MS * sizeof(float));
    cudaMemset(cuda_signalQ1, 0, SAMPLES_PER_MS * sizeof(float));


    cudaMemcpy(cuda_signalI1, signalI1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_signalQ1, signalQ1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);

    float max_cross = 0 ;
    float max_freq = 0 ;
    int max_lag = 0 ;
    std::complex<float> max_sum = std::complex<float>(0.0f, 0.0f);
    float IF = 0;//10.23e6; // 1.023 MHz * 10
    for (lag = 0; lag < SAMPLES_PER_MS; lag += 3)
    // lag = 1230;
    {

        for (freqShiftHz = IF-5000; freqShiftHz <= IF + 5000; freqShiftHz += 250)
        // freqShiftHz = -250;//3250;
        {

            gpu_freq_shift_correlate<<<(SAMPLES_PER_MS/256)+1, 256>>>(0, cuda_signalI1, cuda_signalQ1, cuda_goldCode,  freqShiftHz, lag, cuda_output);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
                exit(1); // If CUDA fails, there is nothing we can do
            }


            cudaMemcpy(output, cuda_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            // for (int i = 0; i <  OUTPUT_SIZE/2; i++) {
            //     printf("(%f,%f)", output[2*i], output[2*i+1]);
            // }

            // exit(1);

            std::complex<float> sum = std::complex<float>(0.0f, 0.0f);
            for (size_t i = 0; i < BLOCK_SIZE; i++) {
                sum += std::complex<float>(output[2*i], output[2*i+1]);
            }
            float realSum = std::abs(sum);

            if (realSum > max_cross) {
                max_cross = realSum;
                max_freq = freqShiftHz;
                max_lag = lag;
                max_sum = sum;
                // printf( "Sat #%d freqShiftHz:%f Lag:%d Cross:%d\n", i, freqShiftHz, lag, cross_cuda);
            }
            // printf( "Lag: %d  FreqShiftHz:%f Cross:(%f,%f)\n", lag, freqShiftHz, sum.imag(), sum.real());
            // printf("%f,", realSum);
        }
        // printf("\n");/
    }
    printf("max cross:%f max freq:%f max lag:%d chips:%d\n", max_cross, max_freq, max_lag, max_lag/SAMPLES_PER_CHIP);
    cudaFree(cuda_output);
    cudaFree(cuda_signalI1);
    cudaFree(cuda_signalQ1);
    cudaFree(cuda_goldCode);

    return max_sum;// (int)realSum;


}


__global__ void gpu_correlate(float* cuda_signalI1, float* cuda_signalQ1, float* cuda_signalI2, float* cuda_signalQ2, int lag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_dataI[256];
    __shared__ float shared_dataQ[256];

    float real1 = cuda_signalI1[i] * cuda_signalI2[(i + lag)%SAMPLES_PER_MS] - cuda_signalQ1[i] * cuda_signalQ2[(i + lag)%SAMPLES_PER_MS];
    float imag1 = cuda_signalI1[i] * cuda_signalQ2[(i + lag)%SAMPLES_PER_MS] + cuda_signalQ1[i] * cuda_signalI2[(i + lag)%SAMPLES_PER_MS];

    shared_dataI[threadIdx.x] = real1;
    shared_dataQ[threadIdx.x] = imag1;

    __syncthreads();

    // Reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_dataI[threadIdx.x] += shared_dataI[threadIdx.x + stride];
            shared_dataQ[threadIdx.x] += shared_dataQ[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        cuda_signalI1[blockIdx.x] = shared_dataI[0];
        cuda_signalQ1[blockIdx.x] = shared_dataQ[0];
    }


}


float crossCorrelationCUDA(
    const std::vector<std::complex<float>>& signal1,
    const std::vector<std::complex<float>>& signal2,
    int lag) {



    float signalI1[SAMPLES_PER_MS];
    float signalQ1[SAMPLES_PER_MS];
    float signalI2[SAMPLES_PER_MS];
    float signalQ2[SAMPLES_PER_MS];

    for (size_t i = 0; i < SAMPLES_PER_MS; ++i) {
        signalI1[i] = signal1[i].real();
        signalQ1[i] = signal1[i].imag();
        signalI2[i] = signal2[i].real();
        signalQ2[i] = signal2[i].imag();
    }



    float *cuda_signalI1;
    float *cuda_signalQ1;
    float *cuda_signalI2;
    float *cuda_signalQ2;


    cudaMalloc(&cuda_signalI1, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&cuda_signalQ1, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&cuda_signalI2, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&cuda_signalQ2, BLOCK_SIZE * sizeof(float));

    cudaMemset(cuda_signalI1, 0, BLOCK_SIZE * sizeof(float));
    cudaMemset(cuda_signalQ1, 0, BLOCK_SIZE * sizeof(float));
    cudaMemset(cuda_signalI2, 0, BLOCK_SIZE * sizeof(float));
    cudaMemset(cuda_signalQ2, 0, BLOCK_SIZE * sizeof(float));


    cudaMemcpy(cuda_signalI1, signalI1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_signalQ1, signalQ1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_signalI2, signalI2, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_signalQ2, signalQ2, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);


    gpu_correlate<<<BLOCK_SIZE, 256>>>(cuda_signalI1, cuda_signalQ1, cuda_signalI2, cuda_signalQ2, lag);

    cudaMemcpy(signalI1, cuda_signalI1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(signalQ1, cuda_signalQ1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyDeviceToHost);

    std::complex<float> sum = std::complex<float>(0.0f, 0.0f);
    for (size_t i = 0; i < BLOCK_SIZE; i++) {
        sum += std::complex<float>(signalI1[i], signalQ1[i]);
    }
    float realSum = std::abs(sum);


    cudaFree(cuda_signalI1);
    cudaFree(cuda_signalQ1);
    cudaFree(cuda_signalI2);
    cudaFree(cuda_signalQ2);

    return realSum;

}


// CUDA Kernel function that runs on the GPU
// __global__ specifies that this function is a kernel and can be called from the CPU
__global__ void gpu_resample(float *cuda_fGoldCode, float *cuda_baseBandI, float *cuda_baseBandQ, float frequencyHz)
{


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float inPhase = cuda_fGoldCode[(i/SAMPLES_PER_CHIP)%1023] == 1 ? 1 : -1;
    cuda_baseBandI[i] = inPhase;
    cuda_baseBandQ[i] = 0.0f;

    float carrier_phase = 2.0f * M_PI * frequencyHz * ((float)i/SAMPLES_PER_SECOND);//(i / 10000.0f);
    cuda_baseBandQ[i] = cuda_baseBandI[i] * sin(carrier_phase);
    cuda_baseBandI[i] *= cos(carrier_phase);

}


// // Convert gold codes to baseband (complex-valued signal)
// std::vector<std::complex<float>> frequencyShiftAndCorelationCUDA(
//     const std::vector<int>& goldCode, float frequencyHz , const std::vector<std::complex<float>>& inputSignal) {

//     float signalI1[SAMPLES_PER_MS];
//     float signalQ1[SAMPLES_PER_MS];

//     for (size_t i = 0; i < SAMPLES_PER_MS; ++i) {
//         signalI1[i] = inputSignal[i].real();
//         signalQ1[i] = inputSignal[i].imag();
//     }



//     float *cuda_signalI1;
//     float *cuda_signalQ1;
//     // float *cuda_signalI2;
//     // float *cuda_signalQ2;


//     cudaMalloc(&cuda_signalI1, BLOCK_SIZE * sizeof(float));
//     cudaMalloc(&cuda_signalQ1, BLOCK_SIZE * sizeof(float));
//     // cudaMalloc(&cuda_signalI2, BLOCK_SIZE * sizeof(float));
//     // cudaMalloc(&cuda_signalQ2, BLOCK_SIZE * sizeof(float));

//     cudaMemset(cuda_signalI1, 0, BLOCK_SIZE * sizeof(float));
//     cudaMemset(cuda_signalQ1, 0, BLOCK_SIZE * sizeof(float));
//     // cudaMemset(cuda_signalI2, 0, BLOCK_SIZE * sizeof(float));
//     // cudaMemset(cuda_signalQ2, 0, BLOCK_SIZE * sizeof(float));


//     cudaMemcpy(cuda_signalI1, signalI1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(cuda_signalQ1, signalQ1, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);
//     // cudaMemcpy(cuda_signalI2, signalI2, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);
//     // cudaMemcpy(cuda_signalQ2, signalQ2, SAMPLES_PER_MS * sizeof(float), cudaMemcpyHostToDevice);

//     gpu_resample<<<(SAMPLES_PER_MS/256)+1, 256>>>(cuda_fGoldCode, cuda_baseBandI, cuda_baseBandQ, frequencyHz);

// }


// Convert gold codes to baseband (complex-valued signal)
std::vector<std::complex<float>> resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(
    const std::vector<int>& goldCode, float frequencyHz ) {


    float baseBandI[SAMPLES_PER_MS];
    float baseBandQ[SAMPLES_PER_MS];
    float fGoldCode[1023];

    float *cuda_fGoldCode;
    float *cuda_baseBandI;
    float *cuda_baseBandQ;

    std::vector<std::complex<float>> baseband(SAMPLES_PER_MS);
    size_t n = baseband.size();

    for (size_t i = 0; i < 1023; ++i) {
        fGoldCode[i] = static_cast<float>(goldCode[i]);
    }



    cudaMalloc(&cuda_fGoldCode, 1023 * sizeof(float));
    cudaMalloc(&cuda_baseBandI, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&cuda_baseBandQ, BLOCK_SIZE * sizeof(float));


    // cudaMemset(cuda_baseBandI, 0, BLOCK_SIZE * sizeof(float));
    // cudaMemset(cuda_baseBandQ, 0, BLOCK_SIZE * sizeof(float));
    // memset(cuda_baseBandI, 0, BLOCK_SIZE * sizeof(float));
    // memset(cuda_baseBandQ, 0, BLOCK_SIZE * sizeof(float));

    cudaMemcpy(cuda_fGoldCode, fGoldCode, 1023 * sizeof(float), cudaMemcpyHostToDevice);


/// CUDA WORLD STARTS HERE


    // Host code example
    auto start = cuda::std::chrono::system_clock::now();


    gpu_resample<<<BLOCK_SIZE, 256>>>(cuda_fGoldCode, cuda_baseBandI, cuda_baseBandQ, frequencyHz);

    // ... some operations ...
    auto end = cuda::std::chrono::system_clock::now();
    cuda::std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    /// CUDA WORLD ENDS HERE

    cudaMemcpy(baseBandI, cuda_baseBandI, SAMPLES_PER_MS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(baseBandQ, cuda_baseBandQ, SAMPLES_PER_MS * sizeof(float), cudaMemcpyDeviceToHost);

   for (size_t i = 0; i < n; ++i) {
        baseband[i] = std::complex<float>( baseBandI[i] , baseBandQ[i]);
    }


    cudaFree(cuda_baseBandI);
    cudaFree(cuda_baseBandQ);
    cudaFree(cuda_fGoldCode);

    return baseband;
}




// Function to convert compute capability to the number of cores per SM
int coresPerSM(int major, int minor) {
    // Refer to NVIDIA documentation for the most up-to-date numbers
    switch (major) {
        case 2: return (minor == 1) ? 48 : 32; // Fermi
        case 3: return 192; // Kepler
        case 5: return 128; // Maxwell
        case 6: return (minor == 0) ? 64 : 128; // Pascal
        case 7: return 64; // Volta and Turing
        case 8: return (minor == 0) ? 64 : 128; // Ampere (8.6, 8.9)
        case 9: return 128; // Hopper
        case 10: return 128; // Blackwell
        default: return 0; // Unknown architecture
    }
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xa3, 128},
      {0xb0, 128},
      {0xc0, 128},
      {0xc1, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}




void print_cuda_props() {
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);

    // if (deviceCount == 0) {
    //     printf("No CUDA-capable devices found.\n");
    //     return;
    // }

    // for (int dev = 0; dev < deviceCount; dev++) {
    //     cudaDeviceProp prop;
    //     cudaGetDeviceProperties(&prop, dev);

    //     int cores = coresPerSM(prop.major, prop.minor) * prop.multiProcessorCount;
    //     printf("Device %d: \"%s\"\n", dev, prop.name);
    //     printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    //     printf("  Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    //     printf("  Multiprocessors (SMs): %d\n", prop.);

    //     printf("  Total CUDA Cores: %d\n", cores);
    // }

    // printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int         deviceCount = 0;
    cudaError_t error_id    = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    }
    else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000,
               (driverVersion % 100) / 10,
               runtimeVersion / 1000,
               (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(msg,
                  sizeof(msg),
                  "  Total amount of global memory:                 %.0f MBytes "
                  "(%llu bytes)\n",
                  static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                  (unsigned long long)deviceProp.totalGlobalMem);
#else
        snprintf(msg,
                 sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes "
                 "(%llu bytes)\n",
                 static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                 (unsigned long long)deviceProp.totalGlobalMem);
#endif
        printf("%s", msg);

        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f "
               "GHz)\n",
               clockRate * 1e-3f,
               clockRate * 1e-6f);
#if CUDART_VERSION >= 5000
        int memoryClockRate;
#if CUDART_VERSION >= 13000
        cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, dev);
#else
        memoryClockRate = deviceProp.memoryClockRate;
#endif
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }

#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the
        // CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }

#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
               "%d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D,
               deviceProp.maxTexture2D[0],
               deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0],
               deviceProp.maxTexture3D[1],
               deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0],
               deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
               "layers\n",
               deviceProp.maxTexture2DLayered[0],
               deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %zu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %zu bytes\n", deviceProp.textureAlignment);
        int gpuOverlap;
        cudaDeviceGetAttribute(&gpuOverlap, cudaDevAttrGpuOverlap, dev);
        printf("  Concurrent copy and kernel execution:          %s with %d copy "
               "engine(s)\n",
               (gpuOverlap ? "Yes" : "No"),
               deviceProp.asyncEngineCount);
        int kernelExecTimeout;
        cudaDeviceGetAttribute(&kernelExecTimeout, cudaDevAttrKernelExecTimeout, dev);
        printf("  Run time limit on kernels:                     %s\n", kernelExecTimeout ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
               deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Managed Memory:                %s\n", deviceProp.managedMemory ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               deviceProp.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");
        // The property cooperativeMultiDeviceLaunch is deprecated in CUDA 13.0
#if CUDART_VERSION < 13000
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
#endif
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               deviceProp.pciDomainID,
               deviceProp.pciBusID,
               deviceProp.pciDeviceID);

        const char *sComputeMode[] = {"Default (multiple host threads can use ::cudaSetDevice() with device "
                                      "simultaneously)",
                                      "Exclusive (only one host thread in one process is able to use "
                                      "::cudaSetDevice() with this device)",
                                      "Prohibited (no host thread can use ::cudaSetDevice() with this "
                                      "device)",
                                      "Exclusive Process (many threads in one process is able to use "
                                      "::cudaSetDevice() with this device)",
                                      "Unknown",
                                      NULL};
        int         computeMode;
        cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, dev);
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (deviceCount >= 2) {
        cudaDeviceProp prop[64];
        int            gpuid[64]; // we want to find the first two GPUs that can support P2P
        int            gpu_p2p_count = 0;

        for (int i = 0; i < deviceCount; i++) {
            cudaGetDeviceProperties(&prop[i], i);
            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                // on Windows (64-bit), the Tesla Compute Cluster driver for windows
                // must be enabled to support this
                && prop[i].tccDriver
#endif
            ) {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int can_access_peer;

        if (gpu_p2p_count >= 2) {
            for (int i = 0; i < gpu_p2p_count; i++) {
                for (int j = 0; j < gpu_p2p_count; j++) {
                    if (gpuid[i] == gpuid[j]) {
                        continue;
                    }
                    cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]);
                    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                           prop[gpuid[i]].name,
                           gpuid[i],
                           prop[gpuid[j]].name,
                           gpuid[j],
                           can_access_peer ? "Yes" : "No");
                }
            }
        }
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char        cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
    snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#endif
    sProfileString += cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
    snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#endif
    sProfileString += cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
    sProfileString += cTemp;
    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

    printf("Result = PASS\n");

}
