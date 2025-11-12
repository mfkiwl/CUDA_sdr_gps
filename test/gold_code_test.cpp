#include "gtest/gtest.h" // Include the Google Test header
#include "C_A__code_generator.h"


int first_ten_bits_binary_to_int(vector<int> input)
{
    int n = 0;
    for (int i = 0 ; i < 10; i++) {
        n *= 2;
        n += input[i];
    }
    return n;
}

vector<string> gold_codes_initials = {"01440","01620","01710","01744","01133","01455","01131","01454","01626","01504","01642","01750","01764","01772","01775","01776","01156","01467","01633","01715","01746","01763","01063","01706","01743","01761","01770","01774","01127","01453","01625","01712"};

TEST(GoldCodeTest, CheckAllSatelites) {

    vector<int> output(1023);
    for (int i = 0 ;i < 32 ; i++){
        CA_generator ca;
        ca.get_gold_code_sequence(i, output);
        printf("Sat #%d Seq:%o\n",i, first_ten_bits_binary_to_int(output));
        int gold_code_initials_decimal_value = strtol(gold_codes_initials[i].c_str(), NULL, 8);

        ASSERT_EQ(first_ten_bits_binary_to_int(output), gold_code_initials_decimal_value);

    }
}

// TEST autocorelation

// TEST crosscorelation

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv); // Initialize Google Test
    return RUN_ALL_TESTS();                // Run all defined tests
}