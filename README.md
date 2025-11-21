# GPS emulator (transmitor) using CUDA 
## About
This is a project to create GPS emulator using HackerRF ONE and CUDA. 

## Capturing real data using HackeRF one
Use command:
hackrf_transfer -r wave.dat -f 1575420000 -p 1 -a 1 -l 40 -g 40 -s 10000000

## Installation: <br>
cd build <br>
cmake .. <br>
make  <br>
