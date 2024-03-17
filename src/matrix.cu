// Runs the matrix multiplication
//
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "kernels/all_kernels.cuh"
#include "matrix.h"

using T = half;
using Accum = float;

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}

template <typename T, typename accum_type>
void run(int id, size_t M, size_t N, size_t K, T *A, T *B, T *C,
         accum_type alpha, accum_type beta) {
    if (id == 0) {
        cublasReference(A, B, C, M, N, K, alpha, beta);
    } else if (id == 1) {
        simpleWMMAMatrix<16, 16, 16>(A, B, C, M, N, K, alpha, beta);
    } else if (id == 2) {
        BasicGEMM(A, B, C, M, N, K, alpha, beta);
    } else if (id == 3) {
        Buffering(A, B, C, M, N, K, alpha, beta);
    } else if (id == 4) {
        DoubleBuffering(A, B, C, M, N, K, alpha, beta);
    } else {
        throw std::runtime_error("Kernel Id not known");
    }
}
const std::map<int, std::string> kernel_mapping = {{0, "CublasRefernece"},
                                                   {1, "Naive"},
                                                   {2, "BasicGEMM"},
                                                   {3, "Buffering"},
                                                   {4, "DoubleBuffering"}};

int main(int argc, char **argv) {
    if (argc != 2) {
        throw std::runtime_error("Kernel id not selected");
    }
    int kernel_id = std::stoi(argv[1]);
    const Accum alpha{1.0}, beta{0.0};
    constexpr size_t M{4096};
    constexpr size_t N{4096};
    constexpr size_t K{4096};
    Matrix<T> A(M, K);
    Matrix<T> B(K, N);
    Matrix<T> D(M, N);
    Matrix<T> Dref(M, N);
    random_fill_and_sync(A);
    random_fill_and_sync(B);
    fill_and_sync(D, (T)0.0);
    fill_and_sync(Dref, (T)0.0);
    cudaDeviceSynchronize();
    std::vector<int> kernel_ids;
    if (kernel_id == -1) {
        std::cout << "runs with all ids" << std::endl;
        kernel_ids = {0, 2, 3, 4};
    } else {
        kernel_ids = {kernel_id};
    };
    for (auto kernel_id : kernel_ids) {
        std::cout << "\n Kernel: " << kernel_mapping.at(kernel_id) << std::endl;
        cudaEvent_t beg{nullptr}, end{nullptr};
        run<T, Accum>(0, M, N, K, A.device_ptr(), B.device_ptr(),
                      Dref.device_ptr(), alpha, beta);
        Dref.sync(cudaMemcpyDeviceToHost);
        run<T, Accum>(kernel_id, M, N, K, A.device_ptr(), B.device_ptr(),
                      D.device_ptr(), alpha, beta);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        D.sync(cudaMemcpyDeviceToHost);
        bool equal = compareMatrices(D, Dref);
        //std::cout << "The  matrices are equal " << equal << std::endl;
        if (!equal) {
            throw std::runtime_error("Not equal");
        }
        float elapsed_time{0};
        cudaEventCreate(&beg);
        cudaEventCreate(&end);
        constexpr size_t n{50};
        cudaEventRecord(beg);
        cudaEventSynchronize(beg);
        for (int j = 0; j < n; ++j) {
            run<T, Accum>(kernel_id, M, N, K, A.device_ptr(), B.device_ptr(),
                          D.device_ptr(), alpha, beta);
        };
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        const double flops{2 * std::pow(double(M), 3) * 1e-9};
        //std::cout << "total flops : " << flops << std::endl;
        printf(
            "Average elapsed time: (%7.6f) ms, performance: (%7.1f) GFLOPS. "
            "size: "
            "(%ld).\n",
            elapsed_time / n, (n * flops * 1000) / (double)elapsed_time, M);
        fflush(stdout);
    }
}
