# GPS emulator (transmitor) using CUDA
## About
This is a project to create GPS emulator using HackerRF ONE and CUDA.

## Capturing real data using HackeRF one
Use command:
hackrf_transfer -r wave.dat -f 1575420000 -p 1 -a 1 -l 40 -g 40 -s 10230000

## Installation: <br>
cd build <br>
cmake .. <br>
make  <br>


## Test:
If you don't have HackRF to check you can use simulation create GPS IQ samples with GPS-SDR-SIM project. 
Compile it and run:
./gps-sdr-sim -e brdc0010.22n -o gps_sim_data.raw -s 10230000 -b 8


## Troubleshuting:
If CUDA kernel does not launch, or if there is a launch failure, try:
  sudo rmmod nvidia_uvm
  sudo modprobe nvidia_uvm

