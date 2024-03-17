
#include <cuda_runtime.h>
#include <mma.h>
#include <sys/cdefs.h>

#include <stdexcept>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include <type_traits>

#include "load_shared.cuh"
#include "utils.cuh"
#include "matrix.h"

using T = half;
using Accum = float;
constexpr size_t CUDA_VECTORIZED_BITS_LOAD = 128;

template <class Threadblock, class Warp, class WMMAblock>
__global__ void WarpCompute(half *__restrict__ A, half* __restrict__ B,  half *C,
                            size_t M, size_t N, size_t K) {
    // Determin block dim. Kernel does not deal with spilling
    assert(M % Threadblock::kM == 0);
    assert(N % Threadblock::kN == 0);
    assert(K % Threadblock::kK == 0);
    const size_t threadblock_rows = M / Threadblock::kM;
    const size_t threadblock_cols = N / Threadblock::kN;
    const size_t num_blocks = gridDim.x * gridDim.y * gridDim.z;
    assert(threadblock_rows * threadblock_cols == num_blocks);

    static_assert(Threadblock::kM >= Warp::kM);
    static_assert(Threadblock::kN >= Warp::kN);
    static_assert(Threadblock::kK >= Warp::kK);

    constexpr size_t warpRows = Threadblock::kM / Warp::kM;
    constexpr size_t warpCols = Threadblock::kN / Warp::kN;
    const size_t num_warps = blockDim.x * blockDim.y * blockDim.z / warpSize;
    assert(warpRows * warpCols == num_warps);
    constexpr size_t offset = 8;

    __shared__ half As[Threadblock::kM][Threadblock::kK + offset];
    //__shared__ half Bs[Threadblock::kK][Threadblock::kN];

    const size_t total_threadId = threadIdx.x;
    const size_t thread_num = blockDim.x;
    // constexpr size_t n_gld = 128 / (sizeof(half) * 8);  // bytes to bits
    constexpr size_t n_gld =
        CUDA_VECTORIZED_BITS_LOAD / (sizeof(half) * 8);  // bytes to bits

    const size_t stride_a = thread_num * n_gld / Threadblock::kK;
    const size_t stride_b = thread_num * n_gld / Threadblock::kN;

    const size_t warp_id = threadIdx.x / warpSize;

    const size_t warp_x = warp_id / warpCols;
    const size_t warp_y = warp_id % warpRows;
    const size_t AsWarpOffset = warp_y * Warp::kM;
    const size_t BsWarpOffset = warp_x * Warp::kN;

    constexpr size_t threadRows = Warp::kM / WMMAblock::kM;
    constexpr size_t threadCols = Warp::kN / WMMAblock::kN;

    C = &C[blockIdx.y * Threadblock::kM * N + blockIdx.x * Threadblock::kN +
           warp_y * Warp::kM * N + warp_x * Warp::kN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMAblock::kM, WMMAblock::kN,
                           WMMAblock::kK, half, nvcuda::wmma::row_major>
        A_frag[threadRows];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMAblock::kM, WMMAblock::kN,
                           WMMAblock::kK, half, nvcuda::wmma::row_major>
        B_frag[threadRows];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMAblock::kM,
                           WMMAblock::kN, WMMAblock::kK, half>
        C_frag[threadRows][threadRows];
    for (size_t i = 0; i < threadRows; ++i) {
        for (size_t j = 0; j < threadRows; ++j) {
            nvcuda::wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    for (size_t block = 0; block < K; block += Threadblock::kK) {
        LoaderA.load();
        LoaderB.load();
        LoaderA.next(Threadblock::kK);
        LoaderB.next(Threadblock::kK * N);
        __syncthreads();
#pragma unroll
        for (size_t bk = 0; bk < Threadblock::kK; bk += WMMAblock::kK) {
#pragma unroll
            for (size_t i = 0; i < threadRows; ++i) {
                // size_t A_iter =
                // AsWarpOffset + bk + i * WMMAblock::kM * Threadblock::kK;
                nvcuda::wmma::load_matrix_sync(
                    A_frag[i], &As[AsWarpOffset + i * WMMAblock::kM][bk],
                    Threadblock::kK + offset);
            }
#pragma unroll
            for (size_t i = 0; i < threadRows; ++i) {
                // size_t B_iter =
                // BsWarpOffset + bk * Threadblock::kN + i * WMMAblock::kN;
                nvcuda::wmma::load_matrix_sync(
                    B_frag[i], &Bs[bk][BsWarpOffset + i * WMMAblock::kN],
                    Threadblock::kM);
            }
#pragma unroll
            for (size_t i = 0; i < threadRows; ++i) {
#pragma unroll
                for (size_t j = 0; j < threadRows; j++) {
                    nvcuda::wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j],
                                           C_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (size_t i = 0; i < threadRows; ++i) {
#pragma unroll
        for (size_t j = 0; j < threadCols; ++j) {
            nvcuda::wmma::store_matrix_sync(
                &C[i * WMMAblock::kM * N + j * WMMAblock::kK], C_frag[i][j], N,
                nvcuda::wmma::mem_row_major);
        }
    }
}

// template <typename T, typename accum_type>
void WarpedSharedUnrollVectorize(half *A, half* B, half *C, size_t M, size_t N,
                                 size_t K, float alpha, float beta) {
    using tb = Block<128, 128, 32>;
    // Cutlass has often 256, 128, 64.... for sm80
    using wb = Block<32, 32, 32>;

    using wmma_block = Block<16, 16, 16>;
    assert(M % tb::kM == 0);
    assert(N % tb::kN == 0);
    static_assert(tb::kM % wb::kM == 0);
    static_assert(tb::kN % wb::kN == 0);
    dim3 gdim(M / tb::kM, N / tb::kN, 1);
    dim3 bdim(32 * (tb::kM / wb::kM) * (tb::kN / wb::kN), 1, 1);
    WarpCompute<tb, wb, wmma_block><<<gdim, bdim>>>(A, B, C, M, N, K);
}

int main(int argc, char **argv) {
    const Accum alpha{1.0}, beta{0.0};
    constexpr size_t M{128};
    constexpr size_t N{128};
    constexpr size_t K{128};
    Matrix<T> A(M, K);
    Matrix<T> B(K, N);
    Matrix<T> D(M, N);
    random_fill_and_sync(A);
    random_fill_and_sync(B);
    fill_and_sync(D, (T)0.0);
    cudaDeviceSynchronize();
    WarpedSharedUnrollVectorize(A.device_ptr(), B.device_ptr(), D.device_ptr(), M, N, K, alpha,
            beta);
}
