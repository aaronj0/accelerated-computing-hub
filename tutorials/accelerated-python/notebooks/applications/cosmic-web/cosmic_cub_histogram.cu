// SPDX-License-Identifier: Apache-2.0
// CUB's device histogram behind a host function: an existing C++ library called from the session.
//
// samples_ptr points at n int32 bin indices in [0, k_bins) on the device, hist_ptr at
// k_bins int32 counters, which the call overwrites. The pointers and the stream arrive as
// integers, the way CuPy exposes them. CUB sizes its scratch on a first dry run; the
// buffer is kept across calls, so a call after the first is the library's kernels only.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/device/device_histogram.cuh>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

void cub_histogram(unsigned long long samples_ptr, unsigned long long hist_ptr, long long n, int k_bins,
                   unsigned long long stream)
{
    static void* scratch = nullptr;
    static size_t scratch_bytes = 0;
    const int* samples = reinterpret_cast<const int*>(samples_ptr);
    int* hist = reinterpret_cast<int*>(hist_ptr);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    size_t needed = 0;
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(nullptr, needed, samples, hist, k_bins + 1, 0, k_bins, n, s));
    if (needed > scratch_bytes) {
        if (scratch != nullptr) CUDA_CHECK(cudaFree(scratch));
        CUDA_CHECK(cudaMalloc(&scratch, needed));
        scratch_bytes = needed;
    }
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(scratch, scratch_bytes, samples, hist, k_bins + 1, 0, k_bins, n, s));
}
