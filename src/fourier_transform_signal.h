#ifndef FOURIER_TRANSFORM_SIGNAL_H
#define FOURIER_TRANSFORM_SIGNAL_H

// fourier_transform_signal.h
// Simple header-only FFT: run FFT from array of float (real input) to complex output.
// Usage:
//   std::vector<std::complex<float>> freq = ft::fft_from_real(input_ptr, len);
//   // or
//   std::vector<std::complex<float>> freq = ft::fft_from_real(std::vector<float>{...});

#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>

namespace ft
{
    inline void fft_inplace(std::vector<std::complex<float>> &data, bool );
    inline void fft_inplace(std::complex<float> *data, std::size_t n, bool);



    inline std::size_t next_power_of_two(std::size_t n)
    {
        if (n == 0)
            return 1;
        --n;
        for (std::size_t p = 1; p < sizeof(std::size_t) * 8; p <<= 1)
            n |= n >> p;
        return ++n;
    }

    // FFT from complex input (pointer + length). Input is copied and zero-padded to next power of two.
    inline std::vector<std::complex<float>> fft_from_complex(const std::complex<float> *input, std::size_t length)
    {
        std::size_t n = next_power_of_two(length);
        std::vector<std::complex<float>> buf(n);
        if (input)
        {
            for (std::size_t i = 0; i < length; ++i)
                buf[i] = input[i];
        }
        for (std::size_t i = length; i < n; ++i)
            buf[i] = std::complex<float>(0.0f, 0.0f);
        fft_inplace(buf.data(), n, false);
        return buf;
    }

    // Return a new vector containing the complex conjugates of the input array.
    inline std::vector<std::complex<float>> conj(const std::vector<std::complex<float>> &input)
    {
        std::vector<std::complex<float>> out;
        out.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
            out[i] = std::conj(input[i]);
        return out;
    }

    // Overload for std::vector input
    // inline std::vector<std::complex<float>> conj(const std::vector<std::complex<float>> &input)
    // {
    //     return conj(input.empty() ? nullptr : input.data(), input.size());
    // }

    // FFT from complex vector
    inline std::vector<std::complex<float>> fft_from_complex(const std::vector<std::complex<float>> &input)
    {
        return fft_from_complex(input.empty() ? nullptr : input.data(), input.size());
    }

    // Convenience in-place vector overload for fft_inplace
    inline void fft_inplace(std::vector<std::complex<float>> &data, bool inverse = false)
    {
        if (!data.empty())
            fft_inplace(data.data(), data.size(), inverse);
    }

    // In-place iterative Cooley-Tukey FFT (n must be a power of two)
    inline void fft_inplace(std::complex<float> *data, std::size_t n, bool inverse = false)
    {
        if (n == 0)
            return;
        // bit reversal permutation
        std::size_t j = 0;
        for (std::size_t i = 1; i < n; ++i)
        {
            std::size_t bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                std::swap(data[i], data[j]);
        }

        const float PI = 3.14159265358979323846f;
        // Danielson-Lanczos
        for (std::size_t len = 2; len <= n; len <<= 1)
        {
            float angle = 2.0f * PI / static_cast<float>(len) * (inverse ? 1.0f : -1.0f);
            std::complex<float> wlen(std::cos(angle), std::sin(angle));
            for (std::size_t i = 0; i < n; i += len)
            {
                std::complex<float> w(1.0f, 0.0f);
                std::size_t half = len >> 1;
                for (std::size_t k = 0; k < half; ++k)
                {
                    std::complex<float> u = data[i + k];
                    std::complex<float> v = data[i + k + half] * w;
                    data[i + k] = u + v;
                    data[i + k + half] = u - v;
                    w *= wlen;
                }
            }
        }

        if (inverse)
        {
            float inv_n = 1.0f / static_cast<float>(n);
            for (std::size_t i = 0; i < n; ++i)
                data[i] *= inv_n;
        }
    }

    // Compute FFT from real float array to complex vector.
    // If length is not a power of two, input is zero-padded to next power of two.
    inline std::vector<std::complex<float>> fft_from_real(const float *input, std::size_t length)
    {
        std::size_t n = next_power_of_two(length);
        std::vector<std::complex<float>> buf(n);
        for (std::size_t i = 0; i < length; ++i)
            buf[i] = std::complex<float>(input[i], 0.0f);
        for (std::size_t i = length; i < n; ++i)
            buf[i] = std::complex<float>(0.0f, 0.0f);
        fft_inplace(buf.data(), n, false);
        return buf;
    }

    inline std::vector<std::complex<float>> fft_from_real(const std::vector<float> &input)
    {
        return fft_from_real(input.empty() ? nullptr : input.data(), input.size());
    }

    // Inverse transform: complex -> real (returns real time-domain samples)
    // n is inferred from freq.size(); output length equals freq.size()
    inline std::vector<float> ifft_to_real(const std::vector<std::complex<float>> &freq)
    {
        std::size_t n = freq.size();
        std::vector<std::complex<float>> buf(freq.begin(), freq.end());
        fft_inplace(buf.data(), n, true);
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = buf[i].real();
        return out;
    }

} // namespace ft

#endif // FOURIER_TRANSFORM_SIGNAL_H