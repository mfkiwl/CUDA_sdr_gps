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
const int CHIPS_PER_MS = 10230;

// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../test_data/wave_1565190Mhz_samp_1023KHz_sats_11_19.dat";
// const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../../test_data/wave_157542_sam_1023_sats_5_11_12.dat";
const char* FILE_PATH_GPS_IQ_SAMPLE1 = "../test_data/wave_156542_sam_1023_sats_5_11_12.dat";


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

    ASSERT_EQ(10230, cross);
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

        for (int lag = 0 ; lag < 10230 ; lag+=3) {
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

// Disabled test - takes too long to run
TEST(GoldCodeTest, autocorelate_wit_real_samples) {

    GPS_IQ_reader reader;
    reader.open(FILE_PATH_GPS_IQ_SAMPLE1);
    reader.seekSample(0x4000);

    std::vector<std::complex<float>> iq_samples;
    reader.readSamples(10230, iq_samples); // Read 10 ms of IQ samples

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
#if 0
TEST(GoldCodeTest, autocorelate_wit_real_samples_fft_version)
{

    GPS_IQ_reader reader;
    // reader.open("../test_data/gps_iq_sample2.raw");
    reader.open(FILE_PATH_GPS_IQ_SAMPLE1);

    std::vector<std::complex<float>> iq_samples;
    reader.readSamples(10230, iq_samples); // Read 10 ms of IQ samples

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
//     reader.readSamples(10230, iq_samples); // Read 10 ms of IQ samples

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

TEST(CudaAddTest, BasicAddition) {
    const unsigned int size = 1<<15;
    int a[size], b[size], c[size];
    for (unsigned int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    add_gpu(a, b, c, size);

    for (unsigned int i = 0; i < size; ++i) {
        EXPECT_EQ(c[i], a[i] + b[i]);
    }
}


// Main function to run all tests
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv); // Initialize Google Test
    return RUN_ALL_TESTS();                 // Run all defined tests
}