
#include <tuple>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

// for tsts
#include <string>


using namespace std;

class CA_generator
{

public:
    CA_generator() :G1(10,1), G2(10,1) {}
    int  get_gold_code_sequence(int satelite_num, vector<int>& output) {


        // for 1 to 1023
        for (int i = 0 ; i < 10 ; i++)
        {

            // generate one bit of gold code
          int  next_bit = get_data_from_selector(satelite_num) ^ get_data_from_G1();


            // update G1
            G1_cycle();
            // update G2
            G2_cycle();

            output[i] = next_bit;
        }

        return 0;
    }
private:
    void shift_left(vector<int>&  Gx )
    {
        for (int i = Gx.size()-1; i  > 0 ; i--) {
           Gx[i] = Gx[i-1];
        }
    }

    void G1_cycle() {
       int next_cell_0_value = (G1[2]) ^ (G1[9]);
       shift_left(G1);
       G1[0] = next_cell_0_value;
    }

    void G2_cycle() {

       bool next_cell_0_value = (G2[1]) ^ (G2[2]) ^ (G2[5]) ^ (G2[7]) ^ (G2[8]) ^ (G2[9])  ;
       shift_left(G2);
       G2[0] = next_cell_0_value;
    }

    int get_data_from_selector(int satelite_num)
    {
        return G2[get<0>(tap_array[satelite_num])]  ^ G2[get<1>(tap_array[satelite_num])] ;
    }

    int get_data_from_G1()
    {
        return G1[9];
    }



    vector<int> G2;
    vector<int> G1;
    vector<tuple<int, int>> tap_array = {{1,5},{2,6},{3,7},{4,8},{0,8},{1,9},{0,7},{1,8},{2,9},{1,2},{2,3},{4,5},{5,6},{6,7},{7,8},{8,9},{0,3},{1,4},{2,5},{3,6},{4,7},{5,8},{0,2},{3,5},{4,6},{5,7},{6,8},{7,9},{0,5},{1,6},{2,7},{3,8}};
};
