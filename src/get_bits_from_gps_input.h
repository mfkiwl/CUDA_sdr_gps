#ifndef GET_BITS_FROM_GPS_INPUT_H
#define GET_BITS_FROM_GPS_INPUT_H

#include <complex>
#include <vector>
#include <cmath>
#include <numeric>

struct GPSData {
    std::vector<int> nav_bits;      // Navigation bits
    std::vector<float> correlations; // Correlation values per bit period
    float snr;                       // Signal-to-noise ratio
};

// Extract GPS navigation bits from I/Q samples
// Assumes samples are at GPS L1 frequency, ~1023 chips/ms
GPSData getGPSData(const std::vector<std::complex<float>>& iq_samples,
                    const std::vector<int>& gold_code,
                   int samples_per_ms = 4092) // Typical for 4.092 MHz sampling
{
    GPSData gps_data;


    // Process each millisecond of data
    for (size_t ms = 0; ms < iq_samples.size() / samples_per_ms; ++ms) {
        size_t start_idx = ms * samples_per_ms;
        size_t end_idx = start_idx + samples_per_ms;

        // Integrate I/Q samples over 1ms (one code period)
        std::complex<float> integrated(0, 0);
        for (size_t i = start_idx; i < end_idx && i < iq_samples.size(); ++i) {
            integrated += iq_samples[i] * ((gold_code[i%10] > 0)? std::complex<float>(1.0f, 0.0f) : std::complex<float>(-1.0f, 0.0f));
        }

        // Extract bit: positive real part -> 1, negative -> 0
        float real_part = integrated.real();

        printf("Integrated[%ld]: %f + %fi\n", ms, integrated.real(), integrated.imag());
        gps_data.nav_bits.push_back(real_part >= 0 ? 1 : 0);

        gps_data.correlations.push_back(std::abs(integrated));
    }

    // Compute SNR from correlations
    if (!gps_data.correlations.empty()) {
        float mean = std::accumulate(gps_data.correlations.begin(),
                                     gps_data.correlations.end(), 0.0f) / gps_data.correlations.size();
        gps_data.snr = mean / (mean * 0.1f); // Simplified SNR estimate
    }

    return gps_data;
}

#endif // GET_BITS_FROM_GPS_INPUT_H