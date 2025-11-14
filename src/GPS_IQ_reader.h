// GPS_IQ_reader.h
// Simple header-only reader for HackRF One raw IQ files (interleaved 8-bit unsigned I,Q).
// Each sample: [I0,Q0,I1,Q1,...] where I,Q are uint8_t (0..255) with center ~128.
// Converted to std::complex<float> with normalized floats in approximately [-1, 1].

#pragma once
#include <fstream>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <stdexcept>

class GPS_IQ_reader {
public:
    GPS_IQ_reader() = default;

    // Open a raw IQ file. Throws std::runtime_error on failure.
    void open(const std::string& path) {
        close();
        in_.open(path, std::ios::binary);
        if (!in_)
            throw std::runtime_error("Failed to open file: " + path);

        // determine size
        in_.seekg(0, std::ios::end);
        file_size_ = static_cast<std::streamoff>(in_.tellg());
        in_.seekg(0, std::ios::beg);
        if (file_size_ < 0)
            file_size_ = 0;
    }

    // Close the file if open.
    void close() {
        if (in_.is_open()) in_.close();
        file_size_ = 0;
    }

    bool isOpen() const { return in_.is_open(); }

    // Number of IQ samples remaining (each sample = 2 bytes: I+Q).
    std::size_t samplesAvailable() {
        if (!isOpen()) return 0;
        auto cur = in_.tellg();
        if (cur < 0) return 0;
        std::streamoff rem = file_size_ - cur;
        return static_cast<std::size_t>(rem > 0 ? rem / 2 : 0);
    }

    // Read up to num_samples IQ samples and append to out (as complex<float>).
    // Returned value = number of samples actually read.
    // Normalization: (x - 128) / 128.0f  -> approximate range [-1, +1)
    std::size_t readSamples(std::size_t num_samples, std::vector<std::complex<float>>& out) {
        if (!isOpen() || num_samples == 0) return 0;
        std::size_t avail = samplesAvailable();
        std::size_t to_read = (num_samples < avail) ? num_samples : avail;
        if (to_read == 0) return 0;

        std::vector<uint8_t> buf;
        buf.resize(to_read * 2);
        in_.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        std::streamsize actually_read = in_.gcount();
        if (actually_read <= 0) return 0;
        // ensure even number of bytes
        std::size_t samples_read = static_cast<std::size_t>(actually_read) / 2;
        out.reserve(out.size() + samples_read);
        for (std::size_t i = 0; i < samples_read; ++i) {
            uint8_t i_b = buf[2 * i];
            uint8_t q_b = buf[2 * i + 1];
            float fi = (static_cast<int>(i_b) - 128) / 128.0f;
            float fq = (static_cast<int>(q_b) - 128) / 128.0f;
            out.emplace_back(fi, fq);
        }
        return samples_read;
    }

    // Convenience: read entire file into vector<complex<float>>
    std::vector<std::complex<float>> readAll() {
        std::vector<std::complex<float>> out;
        if (!isOpen()) return out;
        // read in chunks to avoid large single allocation
        const std::size_t chunk_samples = 1 << 20; // ~1M samples per chunk
        while (samplesAvailable() > 0) {
            readSamples(chunk_samples, out);
        }
        return out;
    }

    // Seek to sample index (0-based). Throws on failure.
    void seekSample(std::size_t sample_index) {
        if (!isOpen()) throw std::runtime_error("File not open");
        std::streamoff byte_pos = static_cast<std::streamoff>(sample_index) * 2;
        if (byte_pos < 0 || byte_pos > file_size_) throw std::runtime_error("Seek out of range");
        in_.seekg(byte_pos, std::ios::beg);
    }

private:
    std::ifstream in_;
    std::streamoff file_size_ = 0;
};