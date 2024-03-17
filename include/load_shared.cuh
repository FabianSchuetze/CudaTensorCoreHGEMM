// Loads the matrices in threadblocks in shared memory
#pragma once

#include <assert.h>
#include <cuda_runtime_api.h>

// Macros taken from https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master
#define CP_ASYNC_CG(dst, src, Bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(Bytes))

template <typename T, size_t rows, size_t cols, size_t num_vectorized_loads,
          typename SpanTemplate, typename ThreadOffset, size_t stride>
struct SharedLoaderMatrix {
    SpanTemplate &shmem_;
    const T *global_ptr_;
    const ThreadOffset offset_;
    const size_t ld_;
    constexpr static size_t cols_ = cols;
    constexpr static size_t size_ = rows * cols;
    __device__ SharedLoaderMatrix(const T *global_ptr, SpanTemplate &shmem,
                                  size_t threadId, size_t blockOffset,
                                  size_t ld)
        : global_ptr_(global_ptr + blockOffset),
          shmem_(shmem),
          offset_(threadId),
          ld_(ld){};
    __device__ void load(size_t counter) {
        const T* global_idx = offset_.row * ld_ + offset_.col + global_ptr_;
        const size_t shmem_row = counter * rows + offset_.row;
#pragma unroll
        for (size_t row = 0; row < rows; row += stride) {
            const T *src = row * ld_ + global_idx;
            T *dst = &shmem_(shmem_row + row, offset_.col);
            constexpr size_t load_bytes = 16;
            uint32_t pos_in_ss = __cvta_generic_to_shared((int4 *)dst);
            CP_ASYNC_CG(pos_in_ss, src, load_bytes);
        }
    }
    __device__ void load_blocking() {
        const size_t global_idx = offset_.row * ld_ + offset_.col;
#pragma unroll
        for (size_t row = 0; row < rows; row += stride) {
            const T *src = global_ptr_ + row * ld_ + global_idx;
            T *dst = &shmem_(offset_.row + row,
                             offset_.col);  // + row * cols;
                                            // constexpr size_t load_bytes = 16;
            auto t = reinterpret_cast<const int4 *>(src)[0];
            reinterpret_cast<int4 *>(dst)[0] = t;
        }
    }
    __device__ void load() { load(0); }
    __device__ void next(size_t step) { global_ptr_ = global_ptr_ + step; };
};

template <size_t rows, size_t cols, size_t num_vectorized_loads>
struct Index {
    const size_t row{0};
    const size_t col{0};
    __host__ __device__ explicit Index(size_t laneId)
        : row(laneId / (cols / num_vectorized_loads)),
          col((laneId % (cols / num_vectorized_loads)) *
              num_vectorized_loads){};
};
