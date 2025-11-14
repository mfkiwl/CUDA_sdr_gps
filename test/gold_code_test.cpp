#include <string>


#include "gtest/gtest.h" // Include the Google Test header
#include "C_A__code_generator.h"
#include "baseband_generator.h"
#include "GPS_IQ_reader.h"


int first_ten_bits_binary_to_int(vector<int> input)
{
    int n = 0;
    for (int i = 0 ; i < 10; i++) {
        n *= 2;
        n += input[i];
    }
    return n;
}

// This data is taken from IS-GPS-200H Table 3-Ia
vector<string> gold_codes_initials = {"01440","01620","01710","01744","01133","01455","01131","01454","01626","01504","01642","01750","01764","01772","01775","01776","01156","01467","01633","01715","01746","01763","01063","01706","01743","01761","01770","01774","01127","01453","01625","01712"};

TEST(GoldCodeTest, CheckAllSatelites) {

    vector<int> output(1023);
    for (int i = 0 ;i < 32 ; i++){
        CA_generator ca;
        ca.get_gold_code_sequence(i, output);
        // printf("Sat #%d Seq:%o\n",i, first_ten_bits_binary_to_int(output));
        int gold_code_initials_decimal_value = strtol(gold_codes_initials[i].c_str(), NULL, 8);

        ASSERT_EQ(first_ten_bits_binary_to_int(output), gold_code_initials_decimal_value);

    }
}


TEST(GoldCodeTest, autocorelation) {

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


TEST(GoldCodeTest, crosscorelation) {

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

TEST(GoldCodeTest, autocrosscorelation_with_lag) {

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

TEST(GoldCodeTest, autocorelate_wit_real_samples) {

    GPS_IQ_reader reader;
    reader.open("../test_data/gps_iq_sample1.raw");

    std::vector<std::complex<float>> iq_samples;
    reader.readSamples(10230, iq_samples); // Read 10 ms of IQ samples

    int max_cross = 0;
    for (int i = 0 ; i < 32 ; i++){
        vector<int> output1(1023);
        CA_generator ca;
        ca.get_gold_code_sequence(i, output1);

        printf("Processing Sat #%d\n", i);
        BasebandGenerator bg;
        auto baseband_signal1 = bg.resampleCaGoldCodeTOneMilisecondOfBaseband(output1);

        for (int lag = 0 ; lag < 10230 ; lag++) {
            auto cross = (int)bg.crossCorrelation(iq_samples, baseband_signal1, lag);
            if (cross > max_cross) {
                max_cross = cross;
                printf("Sat #%d Lag:%d Cross:%d\n", i, lag, cross);

            }
        }
    }
    // ASSERT_GT(cross, 8000);
}


// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv); // Initialize Google Test
    return RUN_ALL_TESTS();                // Run all defined tests
}