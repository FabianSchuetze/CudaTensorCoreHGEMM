#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <stdexcept>

template <typename T, std::size_t rows, std::size_t cols>
struct Span {
    __host__ __device__ Span(T *buffer_) : buffer(buffer_){};
    __host__ __device__ T &operator()(std::size_t row, std::size_t col) {
        return buffer[cols * row + col];
    }
    __host__ __device__ T *begin() { return buffer; };
    __host__ __device__ T *end() { return buffer + rows * cols; };
    T *buffer;
    static constexpr size_t cols_ = cols;
};

template <typename Register, size_t sharedloader_stride, size_t stride,
          size_t load_steps, size_t repeats>
struct RegisterLoader {
    Register (&register_)[repeats * load_steps];
    half *ptr;
    const half *start_ptr;  // should be const
    __device__ RegisterLoader(half *ptr_, Register (&reg)[repeats * load_steps])
        : ptr(ptr_), register_(reg), start_ptr(ptr_){};

    __device__ void load(size_t register_counter) {
        const size_t offset = register_counter * load_steps;
#pragma unroll
        for (size_t i = 0; i < load_steps; ++i) {
            nvcuda::wmma::load_matrix_sync(
                register_[offset + i], ptr + i * stride, sharedloader_stride);
        }
    }
    __device__ void load() { load(0); }
    __host__ __device__ void step(size_t stepsize) { ptr = ptr + stepsize; }
    __host__ __device__ void reset(size_t position) {
        ptr = const_cast<half *>(start_ptr + position);
    }
    __host__ __device__ void reset() { reset(0); }
};

template <typename RegisterA, typename RegisterB, typename RegisterC,
          size_t rows, size_t cols, size_t repeats>
struct MatMul {
    RegisterA (&register_a_)[repeats * rows];
    RegisterB (&register_b_)[repeats * cols];
    RegisterC (&register_c_)[rows][cols];

    __device__ MatMul(RegisterA (&register_a)[repeats * rows],
                      RegisterB (&register_b)[repeats * cols],
                      RegisterC (&register_c)[rows][cols])
        : register_a_(register_a),
          register_b_(register_b),
          register_c_(register_c){};
    __device__ void compute(size_t offset) {
        const size_t offset_a = rows * offset;
        const size_t offset_b = cols * offset;
#pragma unroll
        for (size_t i = 0; i < rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < cols; j++) {
                nvcuda::wmma::mma_sync(
                    register_c_[i][j], register_a_[offset_a + i],
                    register_b_[offset_b + j], register_c_[i][j]);
            }
        }
    }
    __device__ void compute() { compute(0); }
};

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

template <size_t BM_, size_t BN_, size_t BK_>
struct Block {
    constexpr static size_t kM = BM_;
    constexpr static size_t kN = BN_;
    constexpr static size_t kK = BK_;
};

// Macros taken from https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master
#define CP_ASYNC_CG(dst, src, Bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
