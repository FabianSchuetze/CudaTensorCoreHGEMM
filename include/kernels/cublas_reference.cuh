#pragma once
#include <cublas_v2.h>

#include <cstddef>
template <typename T, typename accum_type>
void cublasReference(T* A, T* B, T* C, size_t M, size_t N, size_t K,
                     accum_type alpha, accum_type beta) {
    cublasStatus_t status{};
    cublasHandle_t handle{};
    cublasCreate(&handle);
    half halpha = (half)alpha;
    half hbeta = (half)beta;
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                          CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F,
                          N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
