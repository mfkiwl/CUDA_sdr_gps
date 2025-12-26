#include <string>
#include <thread>


#include "gtest/gtest.h" // Include the Google Test header
#include "C_A__code_generator.h"
#include "baseband_generator.h"
#include "GPS_IQ_reader.h"
#include "fourier_transform_signal.h"
#include "get_bits_from_gps_input.h"
#include "cuda_lib.cuh"


// const int SAMPLE_RATE_HZ = 10000000; // 10 MHz sample rate
const int CHIPS_PER_MS = 1023*10;

// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../test_data/wave_1565190Mhz_samp_1023KHz_sats_11_19.dat";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../test_data/wave_GPS_L1_samp_10.23MHZ_sats_11_12_20_21";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../test_data/wave_GPS_L1_samp_10.23MHZ_sats_11_12_20_21.dat";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../../test_data/wave_GPS_L1_samp_10.23MHZ_sats_11_12_20_21.dat2";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../test_data/wave_157542_sam_1023_sats_5_11_12.dat";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../test_data/wave_156542_sam_1023_sats_5_11_12.dat";
const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../../test_data/gps_sim_data_10p23MHZSampling.raw"; // Working
// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../../test_data/gps_sim_data_4p092MHzSampling.raw"; // Working
// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../../test_data/data.raw";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/data_p1_a1_l32_g40_b28_s10p23.raw"; BAD
// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/data_p1_a1_l40_g60_b28_s10p23.raw"; BAD
// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/data_p1_a1_l24_g30_b28_s10p23.raw";


// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/real_gps.raw";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/data_p1_a1_l24_g40_b28_s10p23.raw";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/data_p1_a1_l32_g40_b28_s10p23.raw";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/data_p1_a1_l32_g50_b28_s10p23.raw";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/data_p1_a1_l40_g50_b28_s10p23.raw";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 =  "../../test_data/data_p1_a1_l40_g60_b28_s10p23.raw";


int first_ten_bits_binary_to_int(vector<int> input)
{
    int n = 0;
    for (int i = 0; i < 10; i++)
    {
        n *= 2;
        n += input[i];
    }
    return n;
}

// This data is taken from IS-GPS-200H Table 3-Ia
vector<string> gold_codes_initials = {"01440", "01620", "01710", "01744", "01133", "01455", "01131", "01454", "01626", "01504", "01642", "01750", "01764", "01772", "01775", "01776", "01156", "01467", "01633", "01715", "01746", "01763", "01063", "01706", "01743", "01761", "01770", "01774", "01127", "01453", "01625", "01712"};

TEST(GoldCodeTest, CheckAllSatelites)
{

    vector<int> output(1023);
    for (int i = 0; i < 32; i++)
    {
        CA_generator ca;
        ca.get_gold_code_sequence(i, output);
        // printf("Sat #%d Seq:%o\n",i, first_ten_bits_binary_to_int(output));
        int gold_code_initials_decimal_value = strtol(gold_codes_initials[i].c_str(), NULL, 8);

        ASSERT_EQ(first_ten_bits_binary_to_int(output), gold_code_initials_decimal_value);
    }
}

TEST(GoldCodeTest, autocorelation)
{

    vector<int> output1(1023), output2(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(1, output1);
    ca.get_gold_code_sequence(1, output2);

    BasebandGenerator bg;
    auto baseband_signal1 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output1);
    auto baseband_signal2 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output2);

    auto cross = (int)bg.crossCorrelation(baseband_signal1, baseband_signal2, 0);

    ASSERT_EQ(CHIPS_PER_MS, cross);
}

TEST(GoldCodeTest, crosscorelation)
{

    vector<int> output1(1023), output2(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(1, output1);
    ca.get_gold_code_sequence(14, output2);

    BasebandGenerator bg;
    auto baseband_signal1 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output1);
    auto baseband_signal2 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output2);

    auto cross = (int)bg.crossCorrelation(baseband_signal1, baseband_signal2, 0);

    ASSERT_LT(cross, 650);
}

TEST(GoldCodeTest, autocrosscorelation_with_lag)
{

    vector<int> output1(1023), output2(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(1, output1);
    ca.get_gold_code_sequence(1, output2);
    int lag = 500;

    BasebandGenerator bg;
    auto baseband_signal1 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output1);
    auto baseband_signal2 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output2);

    auto cross = (int)bg.crossCorrelation(baseband_signal1, baseband_signal2, lag);

    ASSERT_LT(cross, 650);
}

void run_one_sateliate_in_thread(int i, std::vector<std::complex<float>> iq_samples)
{
    int max_cross = 0;
    vector<int> output1(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(i, output1);

    printf("Processing Sat #%d\n", i);
    BasebandGenerator bg;
    int max_lag = 0;
    float max_freq = 0;
    for (float freq = -5000; freq <= 5000; freq += 500) {
        auto baseband_signal1 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output1, freq );//- 10e6 );

        for (int lag = 0 ; lag < CHIPS_PER_MS ; lag+=3) {
            auto cross = (int)bg.crossCorrelation(iq_samples, baseband_signal1, lag);
            // printf("%d, ", cross);
            if (cross > max_cross) {
                max_cross = cross;
                max_freq = freq;
                max_lag = lag;
                // printf("Sat #%d Lag:%d Cross:%d freq:%f\n", i, lag, cross, freq);
                // Max for now: Sat #0 Lag:350 Cross:832 freq:-300.000000
            }
        }

        // printf("\n");
    }
    printf("MAX: Sat #%d Lag:%d Cross:%d freq:%f\n", i, max_lag, max_cross, max_freq);

}
#if 0

// Disabled test - takes too long to run
TEST(GoldCodeTest, autocorelate_wit_real_samples) {

    GPS_IQ_reader reader;
    reader.open(FILE_PATH_GPS_IQ_SAMPLE1);
    reader.seekSample(0x4000);

    std::vector<std::complex<float>> iq_samples;
    reader.readSamples(CHIPS_PER_MS, iq_samples); // Read 10 ms of IQ samples

    std::vector<std::thread> threads;

    // for (int i = 4; i < 13 ; i++)
    for (int i = 0; i < 16 ; i++)
    // int i = 0;
    {
        threads.emplace_back(run_one_sateliate_in_thread, i, iq_samples);
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    // ASSERT_GT(cross, 8000);
}

TEST(GoldCodeTest, autocorelate_wit_real_samples_fft_version)
{

    GPS_IQ_reader reader;
    // reader.open("../test_data/gps_iq_sample2.raw");
    reader.open(FILE_PATH_GPS_IQ_SAMPLE1);

    std::vector<std::complex<float>> iq_samples;
    reader.readSamples(CHIPS_PER_MS, iq_samples); // Read 10 ms of IQ samples

    // for (int i = 4; i < 32; i++)
    //Sat #11 Lag:5032 Cross:5335 freq:-500.000000
    //Sat #4 Lag:587 Cross:5122 freq:-500.000000


    int i =11;
    {
        vector<int> output1(1023);
        CA_generator ca;
        ca.get_gold_code_sequence(i, output1);

        // printf("Processing Sat #%d\n", i);
        BasebandGenerator bg;
        int max_cross = 0;
        int max_lag = 0;
        float max_freq = 0;

        float freq = -500;
        //for (float freq = -5000; freq <= 5000; freq += 500)
        {
            auto baseband_signal1 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output1, freq);
            auto baseband_signal1_fft = ft::fft_from_complex(baseband_signal1);
            baseband_signal1_fft = ft::conj(baseband_signal1_fft);

            auto iq_samples_fft = ft::fft_from_complex(iq_samples);

            vector<complex<float>> cross = bg.crossCorrelation(iq_samples_fft, baseband_signal1_fft);


            // ft::fft_inplace(cross, true);
            // auto cross_ifft = cross;
            auto cross_ifft = ft::ifft_to_real(cross);
            // printf("cross_if ft size:%zu\n", cross_ifft.size());

            for (size_t lag = 0; lag < cross_ifft.size(); lag++)
            {
                int cross_value = std::abs(cross_ifft[lag]);
                if (cross_value > max_cross)
                {
                    max_cross = cross_value;
                    max_lag = lag;
                    max_freq = freq;

                    printf("Sat #%d Lag:%zu Cross:%d freq:%f\n", i, lag, cross_value, freq);
                    // Max for now: Sat #0 Lag:350 Cross:832 freq:-300.000000
                }

            }

        }
        printf("Sat #%d Lag:%d Cross:%d freq:%f\n", i, max_lag, max_cross, max_freq);



    }
    // ASSERT_GT(cross, 8000);
}
#endif

// TEST(GoldCodeTest, get_navigation_bits_From_IQ_samples)
// {

//     GPS_IQ_reader reader;

//     reader.open(FILE_PATH_GPS_IQ_SAMPLE1);

//     std::vector<std::complex<float>> iq_samples;
//     reader.readSamples(CHIPS_PER_MS, iq_samples); // Read 10 ms of IQ samples

//     // for (int i = 0; i < 32; i++)
//     int i = 29;
//     {
//         vector<int> output1(1023);
//         CA_generator ca;
//         ca.get_gold_code_sequence(i, output1);

//         printf("Processing Sat #%d\n", i);
//         BasebandGenerator bg;
//         int max_cross = 0;
//         int max_lag = 0;
//         float max_freq = 0;

//         for (float freq = -1500; freq <= 1500; freq += 50)
//         {
//             auto baseband_signal1 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output1, freq);
//             auto baseband_signal1_fft = ft::fft_from_complex(baseband_signal1);
//             baseband_signal1_fft = ft::conj(baseband_signal1_fft);

//             auto iq_samples_fft = ft::fft_from_complex(iq_samples);

//             vector<complex<float>> cross = bg.crossCorrelation(iq_samples_fft, baseband_signal1_fft);

//             // auto cross_ifft = ft::ifft_to_real(cross);
//             ft::fft_inplace(cross, true);
//             auto cross_ifft = cross;
//             // printf("cross_if ft size:%zu\n", cross_ifft.size());

//             for (size_t lag = 0; lag < cross_ifft.size(); lag++)
//             {
//                 int cross_value = std::abs(cross_ifft[lag]);
//                 if (cross_value > max_cross)
//                 {
//                     max_cross = cross_value;
//                     max_lag = lag;
//                     max_freq = freq;

//                     // printf("Sat #%d Lag:%zu Cross:%d freq:%f\n", i, lag, cross_value, freq);
//                     // Max for now: Sat #0 Lag:350 Cross:832 freq:-300.000000
//                 }

//             }

//         }
//         printf("Sat #%d Lag:%d Cross:%d freq:%f\n", i, max_lag, max_cross, max_freq);

//         // max_freq = 300;

//         GPS_IQ_reader reader2;
//         reader2.open(FILE_PATH_GPS_IQ_SAMPLE1);
//         reader2.seekSample(max_lag);

//         std::vector<std::complex<float>> iq_samples;
//         reader2.readSamples(500*CHIPS_PER_MS, iq_samples); // Read 40 ms of IQ samples

//         auto baseband_signal1 = bg.resampleInputSignalToBaseband(iq_samples, max_freq);


//         auto gps_data = getGPSData(baseband_signal1, output1, CHIPS_PER_MS);

//         // First 10 bits should be: 1,0,1,0,1,0,1,0,1,0
//         // std::vector<int> expected_bits = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

//         // ASSERT_EQ(gps_data.nav_bits.size(), 40);
//         // for (size_t i = 0; i < expected_bits.size(); i++)
//         // {
//         //     ASSERT_EQ(gps_data.nav_bits[i], expected_bits[i]);
//         // }
//         // ASSERT_GT(gps_data.snr, 5.0f);



//     }
//     // ASSERT_GT(cross, 8000);
// }


TEST(CudaTest, ResampleGoldCodeToBaseband) {
    vector<int> gold_code(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(1, gold_code);


     auto start = std::chrono::system_clock::now();

    // 2. Perform some operations or wait (for demonstration)
    BasebandGenerator bg;
    auto baseband_signal_cpu = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(gold_code, 1000.0f);


    // 3. Get the ending time point
    auto end = std::chrono::system_clock::now();

    // 4. Calculate the difference (results in a duration object)
    std::chrono::system_clock::duration diff = end - start;

    // For floating-point seconds:
    std::chrono::duration<double> seconds_double = end - start;
    // std::cout << "Time difference in seconds (double): " << seconds_double.count() << std::endl;
    std::cout << "Time difference in seconds (double): " << seconds_double.count() << std::endl;


    auto baseband_signal_gpu = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(gold_code, 1000.0f);

    ASSERT_EQ(baseband_signal_cpu.size(), baseband_signal_gpu.size());

    for (size_t i = 0; i < baseband_signal_cpu.size(); i++) {
        EXPECT_NEAR(baseband_signal_cpu[i].real(), baseband_signal_gpu[i].real(), 1e-5);
        EXPECT_NEAR(baseband_signal_cpu[i].imag(), baseband_signal_gpu[i].imag(), 1e-5);
    }
}

TEST(CudaGoldCodeTest, autocorelation)
{

    vector<int> output1(1023), output2(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(1, output1);
    ca.get_gold_code_sequence(1, output2);

    BasebandGenerator bg;
    auto baseband_signal1 = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(output1, 0);
    auto baseband_signal2 = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(output2, 0);


     for (size_t i = 0; i < 10; i++) { //baseband_signal1.size(); i++) {
        EXPECT_NEAR(baseband_signal2[i].real(), baseband_signal1[i].real(), 1e-5);
        EXPECT_NEAR(baseband_signal2[i].imag(), baseband_signal1[i].imag(), 1e-5);
        // std::cout << "Baseband1[" << i << "]: " << baseband_signal1[i] << " Baseband2[" << i << "]: " << baseband_signal2[i] << std::endl;
    }
    auto cross = (int)bg.crossCorrelation(baseband_signal1, baseband_signal2, 0);

    ASSERT_LE(10220, cross);
}

TEST(CudaGoldCodeTest, autocorelation_with_frequency_shift)
{

    vector<int> output1(1023), output2(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(1, output1);
    ca.get_gold_code_sequence(1, output2);

    BasebandGenerator bg;
    auto baseband_signal1 = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(output1, 250);
    auto baseband_signal2 = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(output2, 250);


     for (size_t i = 0; i < 10; i++) { //baseband_signal1.size(); i++) {
        EXPECT_NEAR(baseband_signal2[i].real(), baseband_signal1[i].real(), 1e-5);
        EXPECT_NEAR(baseband_signal2[i].imag(), baseband_signal1[i].imag(), 1e-5);
        // std::cout << "Baseband1[" << i << "]: " << baseband_signal1[i] << " Baseband2[" << i << "]: " << baseband_signal2[i] << std::endl;
    }
    auto cross = (int)bg.crossCorrelation(baseband_signal1, baseband_signal2, 0);

    ASSERT_LE(10220/2, cross);
}



TEST(CudaGoldCodeTest, autocorelation_with_freq_in_CUDA)
{

    vector<int> output1(1023), output2(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(1, output1);
    ca.get_gold_code_sequence(1, output2);

    BasebandGenerator bg;
    auto baseband_signal1 = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(output1, 250);
    auto baseband_signal2 = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(output2, 250);


     for (size_t i = 0; i < 10; i++) { //baseband_signal1.size(); i++) {
        EXPECT_NEAR(baseband_signal2[i].real(), baseband_signal1[i].real(), 1e-5);
        EXPECT_NEAR(baseband_signal2[i].imag(), baseband_signal1[i].imag(), 1e-5);
        // std::cout << "Baseband1[" << i << "]: " << baseband_signal1[i] << " Baseband2[" << i << "]: " << baseband_signal2[i] << std::endl;
    }
    auto cross = (int)bg.crossCorrelation(baseband_signal1, baseband_signal2, 0);
    auto cross_CUDA = (int)crossCorrelationCUDA(baseband_signal1, baseband_signal2, 0);

    EXPECT_NEAR(cross_CUDA, cross, 30);
}

TEST(CudaGoldCodeTest, autocorelation_with_freq_in_CUDA_and_lag)
{

    vector<int> output1(1023), output2(1023);
    CA_generator ca;
    ca.get_gold_code_sequence(1, output1);
    ca.get_gold_code_sequence(1, output2);

    BasebandGenerator bg;
    auto baseband_signal1 = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(output1, 250);
    auto baseband_signal2 = resampleCaGoldCodeTOneMilisecondOfBasebandCUDA(output2, 250);


     for (size_t i = 0; i < 10; i++) { //baseband_signal1.size(); i++) {
        EXPECT_NEAR(baseband_signal2[i].real(), baseband_signal1[i].real(), 1e-5);
        EXPECT_NEAR(baseband_signal2[i].imag(), baseband_signal1[i].imag(), 1e-5);
        // std::cout << "Baseband1[" << i << "]: " << baseband_signal1[i] << " Baseband2[" << i << "]: " << baseband_signal2[i] << std::endl;
    }
    auto cross = (int)bg.crossCorrelation(baseband_signal1, baseband_signal2, 10);
    auto cross_CUDA = (int)crossCorrelationCUDA(baseband_signal1, baseband_signal2, 10);

    EXPECT_NEAR(cross_CUDA, cross, 30);
}

// TEST(CudaGoldCodeTest, one_resample_and_correlate_in_CUDA)
// {

//     GPS_IQ_reader reader;
//     reader.open(FILE_PATH_GPS_IQ_SAMPLE1);
//     reader.seekSample(0x4000);

//     std::vector<std::complex<float>> iq_samples;
//     reader.readSamples(CHIPS_PER_MS, iq_samples); // Read 10 ms of IQ samples
//     int max_cross = 0;
//     vector<int> output1(1023);
//     CA_generator ca;
//     int i = 1; // test for satelite 1
//     ca.get_gold_code_sequence(i, output1);

//     BasebandGenerator bg;
//     vector<std::complex<float>> local_gold_code = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output1, 0);

//     printf("Processing Sat #%d\n", i);
//     int max_lag = 0;
//     float max_freq = 0;
//     float freq = 1000;
// // for (float freq = -5000; freq <= 5000; freq += 500) {
//     auto baseband_signal1 = bg.resampleInputSignalToBaseband(iq_samples, freq );//- 10e6 );

//     int lag = 0;
//     // for (int lag = 0 ; lag < CHIPS_PER_MS ; lag+=3) {
//     auto cross = (int)bg.crossCorrelation(local_gold_code, baseband_signal1, lag);

//     // Now make it run all in CUDA
//     auto cross_CUDA = (int)crossCorrelationCUDA(local_gold_code, baseband_signal1, lag);


//     EXPECT_EQ(cross_CUDA, cross);

// }

// TEST(CudaGoldCodeTest, resample_and_correlate_in_one_kernel_CUDA)
// {

//     GPS_IQ_reader reader;
//     reader.open(FILE_PATH_GPS_IQ_SAMPLE1);
//     reader.seekSample(0x4000);

//     std::vector<std::complex<float>> iq_samples;
//     reader.readSamples(CHIPS_PER_MS, iq_samples); // Read 10 ms of IQ samples
//     int max_cross = 0;
//     vector<int> goldCode(1023);
//     CA_generator ca;
//     int i = 1; // test for satelite 1
//     ca.get_gold_code_sequence(i, goldCode);
//     float freqShiftHz = 0;//1000;
//     int lag = 0;



//     BasebandGenerator bg;
//     vector<std::complex<float>> local_gold_code = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(goldCode, 0);

//     printf("Processing Sat #%d\n", i);

//     auto baseband_signal1 = bg.resampleInputSignalToBaseband(iq_samples,    freqShiftHz );//- 10e6 );

//     auto cross = (int)bg.crossCorrelation(local_gold_code, baseband_signal1, lag);

//     auto cross_cuda = freq_shift_correlateCUDA(goldCode, freqShiftHz , iq_samples,  lag) ;
//     EXPECT_EQ(cross_cuda, cross);

// }


TEST(CudaGoldCodeTest, findOneSateCUDA)
{

    GPS_IQ_reader reader;
    reader.open(FILE_PATH_GPS_IQ_SAMPLE1);
    reader.seekSample(0x4000);

    std::vector<std::complex<float>> iq_samples;
    reader.readSamples(CHIPS_PER_MS, iq_samples); // Read 10 ms of IQ samples
    int max_cross = 0;
    vector<int> goldCode(1023);
    CA_generator ca;

    printf("Running on file :%s\n", FILE_PATH_GPS_IQ_SAMPLE1);

    for (int i = 0; i < 32; i++)
    // for (int i = 10; i < 14; i++)
    // int i =13;
    {

        // int i = 4; // test for satelite 1
        ca.get_gold_code_sequence(i, goldCode);
        float freqShiftHz = 0;//1000;
        int lag = 0;

        printf("Processing Sat #%d\n", i);
        // for (freqShiftHz = -5000; freqShiftHz <= 5000; freqShiftHz += 500) {

        //     for (lag = 0 ; lag < CHIPS_PER_MS ; lag+=3) {
            for (int samples = 0 ; samples < 3; samples++) {
                reader.readSamples(CHIPS_PER_MS, iq_samples); // Read 10 ms of IQ samples

                auto cross_cuda_complex = freq_shift_correlateCUDA(goldCode, freqShiftHz , iq_samples,  lag) ;
                iq_samples.clear();

                // auto cross_cuda = (int)abs(cross_cuda_complex);
                // if (cross_cuda > max_cross) {
                //     max_cross = cross_cuda;
                //     printf( "Sat #%d freqShiftHz:%f Lag:%d Cross:%d\n", i, freqShiftHz, lag, cross_cuda);
                // }
            }
        // }
    }
}



TEST(CudaGoldCodeTest, runOneSateMultipleChipsLimitedSearchCUDA)
{

    GPS_IQ_reader reader;
    reader.open(FILE_PATH_GPS_IQ_SAMPLE1);
    reader.seekSample(0x4000);

    std::vector<std::complex<float>> iq_samples;
    int max_cross = 0;
    vector<int> goldCode(1023);
    CA_generator ca;

    int satellite_id =10;//=4;
    ca.get_gold_code_sequence(satellite_id, goldCode);
    float freqShiftHz = -4750.000000f;// -2750;
    int lag = 5900;//5184;

    iq_samples.clear();

    printf("Processing Sat #%d\n", satellite_id);
    for (int i = 0;  i < 2000 ; i++) {
        // reader.seekSample(0x4000 + i*CHIPS_PER_MS);
        auto samples_out_num = reader.readSamples(CHIPS_PER_MS, iq_samples); // Read 10 ms of IQ samples

        // for (int i = 0; i < CHIPS_PER_MS; i++) {
        //     float gc = goldCode[(i/10+123)%1023] == 1 ? 1.0f : -1.0f;
        //     // float gc = goldCode[(i/10)%1023] == 1 ? 1.0f : -1.0f;
        //     iq_samples.push_back(std::complex<float>(gc* std::cos(2.0f * 3.14159265f * freqShiftHz * i / 10.23e6f),
        //                                              gc* std::sin(2.0f * 3.14159265f * freqShiftHz * i / 10.23e6f)));
        //                                             // goldCode[(i/10+123)%1023] * std::sin(2.0f * 3.14159265f * freqShiftHz * i / 10.23e6f)));
        // }

        auto cross_cuda = freq_shift_correlateLimitedSearchCUDA(goldCode, freqShiftHz , iq_samples,  lag, i) ;
        iq_samples.clear();
        printf("Sample %d: Cross:(%f, %f) ", i, cross_cuda.real(), cross_cuda.imag());
    }

}


// Main function to run all tests
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv); // Initialize Google Test
    return RUN_ALL_TESTS();                 // Run all defined tests
}