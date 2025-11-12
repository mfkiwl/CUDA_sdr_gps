
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
    int  get_gold_code_sequence(int satelite_num, vector<int>& output) ;

private:
    void shift_left(vector<int>&  Gx );

    void G1_cycle() ;

    void G2_cycle() ;

    int get_data_from_selector(int satelite_num);

    int get_data_from_G1();

    vector<int> G2;
    vector<int> G1;
    vector<tuple<int, int>> tap_array = {{1,5},{2,6},{3,7},{4,8},{0,8},{1,9},{0,7},{1,8},{2,9},{1,2},{2,3},{4,5},{5,6},{6,7},{7,8},{8,9},{0,3},{1,4},{2,5},{3,6},{4,7},{5,8},{0,2},{3,5},{4,6},{5,7},{6,8},{7,9},{0,5},{1,6},{2,7},{3,8}};
};


