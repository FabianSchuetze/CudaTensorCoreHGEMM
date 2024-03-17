#pragma once

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cassert>
#include <cstddef>
#include <stdexcept>

template <size_t BM, size_t BN, size_t BK>
__global__ void simpleWMMAMatrixKernel(half *A, half *B, half *C, size_t M,
                                       size_t N, size_t K, float alpha,
                                       float beta) {
    const size_t warp_col = blockIdx.x * BK;
    const size_t warp_row = blockIdx.y * BN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, BM, BN, BK, half,
                           nvcuda::wmma::row_major>
        A_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, BM, BN, BK, half,
                           nvcuda::wmma::row_major>
        B_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, BM, BN, BK, half> C_frag;
    nvcuda::wmma::fill_fragment(C_frag, 0.0);
    for (size_t k = 0; k < K; k += BK) {
        size_t elea = k + K * warp_row;
        size_t eleb = k * N + warp_col;
        nvcuda::wmma::load_matrix_sync(A_frag, &A[elea], K);
        nvcuda::wmma::load_matrix_sync(B_frag, &B[eleb], N);
        nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    const size_t elec = warp_row * N + warp_col;
    nvcuda::wmma::store_matrix_sync(&C[elec], C_frag, N,
                                    nvcuda::wmma::mem_row_major);
}

template <size_t BM, size_t BN, size_t BK>
void simpleWMMAMatrix(half *A, half *B, half *C, size_t M, size_t N, size_t K,
                      float alpha, float beta) {
    if (M % 32 != 0) {
        throw std::runtime_error("Not divisible by 32");
    }
    if (N % 32 != 0) {
        throw std::runtime_error("Not divisible by 32");
    }
    dim3 gridDim(M / 16, N / 16);
    dim3 blockDim(32);
    simpleWMMAMatrixKernel<BM, BN, BK>
        <<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}
