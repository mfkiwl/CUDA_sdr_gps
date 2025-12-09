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



// CUDA Kernel function that runs on the GPU
// __global__ specifies that this function is a kernel and can be called from the CPU
__global__ void gpu_resample(float *cuda_fGoldCode, float *cuda_baseBandI, float *cuda_baseBandQ, float frequencyHz) {


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // for (size_t i = 0; i < n; ++i)
    {
        float inPhase = cuda_fGoldCode[(i/10)%1023] == 1 ? 1 : -1;
        cuda_baseBandI[i] = inPhase;
        cuda_baseBandQ[i] = 0.0f;
    }


    // Modulate with complex exponential carrier
    // for (size_t i = 0; i < n; ++i)
    {

        //500 * 2.0f * M_PI * (i / 10000.0f);
        // 500 Hz carrier frequency over 1 ms at 10 MHz sample rate
        float carrier_phase = 2.0f * M_PI * frequencyHz * ((float)i/10230000);//(i / 10000.0f);
        cuda_baseBandQ[i] = cuda_baseBandI[i] * sin(carrier_phase);
        cuda_baseBandI[i] *= cos(carrier_phase);
    }

    // cuda_baseBandI[i] +=1.0f;  // Shift Q to make signal non-negative
    // cuda_baseBandQ[i] +=1.0f;  // Shift Q to make signal non-negative
}

// Convert gold codes to baseband (complex-valued signal)
std::vector<std::complex<float>> resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(
    const std::vector<int>& goldCode, float frequencyHz ) {


    static float baseBandI[10230];
    static float baseBandQ[10230];
    static float fGoldCode[1023];

    float *cuda_fGoldCode;
    float *cuda_baseBandI;
    float *cuda_baseBandQ;

    std::vector<std::complex<float>> baseband(10240);
    size_t n = baseband.size();

    for (size_t i = 0; i < 1023; ++i) {
        fGoldCode[i] = static_cast<float>(goldCode[i]);
    }



    cudaMalloc(&cuda_fGoldCode, 1023 * sizeof(float));
    cudaMalloc(&cuda_baseBandI, 10240 * sizeof(float));
    cudaMalloc(&cuda_baseBandQ, 10240 * sizeof(float));


    // cudaMemset(cuda_baseBandI, 0, 10240 * sizeof(float));
    // cudaMemset(cuda_baseBandQ, 0, 10240 * sizeof(float));
    // memset(cuda_baseBandI, 0, 10240 * sizeof(float));
    // memset(cuda_baseBandQ, 0, 10240 * sizeof(float));

    cudaMemcpy(cuda_fGoldCode, fGoldCode, 1023 * sizeof(float), cudaMemcpyHostToDevice);


/// CUDA WORLD STARTS HERE


    // Host code example
    auto start = cuda::std::chrono::system_clock::now();


    gpu_resample<<<(10230/256)+1, 256>>>(cuda_fGoldCode, cuda_baseBandI, cuda_baseBandQ, frequencyHz);

    // ... some operations ...
    auto end = cuda::std::chrono::system_clock::now();
    cuda::std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    /// CUDA WORLD ENDS HERE

    cudaMemcpy(baseBandI, cuda_baseBandI, 10230 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(baseBandQ, cuda_baseBandQ, 10230 * sizeof(float), cudaMemcpyDeviceToHost);

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
