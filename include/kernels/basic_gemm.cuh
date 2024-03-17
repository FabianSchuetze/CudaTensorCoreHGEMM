#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <sys/cdefs.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "load_shared.cuh"
#include "utils.cuh"

constexpr size_t CUDA_VECTORIZED_BITS_LOAD = 128;

template <class Threadblock, class Warp, class WMMAblock>
__global__ void BasicGEMM(const half *__restrict__ A,
                            const half *__restrict__ B, half *C, size_t M,
                            size_t N, size_t K) {
    //  Determin block dim. Kernel does not deal with spilling
    assert(M % Threadblock::kM == 0);
    assert(N % Threadblock::kN == 0);
    assert(K % Threadblock::kK == 0);

    static_assert(Threadblock::kM >= Warp::kM);
    static_assert(Threadblock::kN >= Warp::kN);
    static_assert(Threadblock::kK >= Warp::kK);

    constexpr size_t warpRows = Threadblock::kM / Warp::kM;
    constexpr size_t warpCols = Threadblock::kN / Warp::kN;
    constexpr size_t num_warps = warpRows * warpCols;
    static_assert(warpRows * warpCols == num_warps);

    constexpr size_t skew = 8;

    extern __shared__ half shmem[];
    using SpanTypeA = Span<half, Threadblock::kM, Threadblock::kK + skew>;
    SpanTypeA As(&shmem[0]);
    using SpanTypeB = Span<half, Threadblock::kK, Threadblock::kN + skew>;
    SpanTypeB Bs(As.end());

    const size_t total_threadId = threadIdx.x;
    constexpr size_t thread_num = num_warps * 32;
    static_assert(thread_num <= 512);  // only up to 512 supported

    // constexpr size_t n_gld = 128 / (sizeof(half) * 8);  // bytes to bits
    constexpr size_t n_gld =
        CUDA_VECTORIZED_BITS_LOAD / (sizeof(half) * 8);  // bytes to bits

    constexpr size_t stride_a = thread_num * n_gld / Threadblock::kK;
    constexpr size_t stride_b = thread_num * n_gld / Threadblock::kN;

    using MemLoaderA = SharedLoaderMatrix<
        half, Threadblock::kM, Threadblock::kK + skew, n_gld, SpanTypeA,
        Index<Threadblock::kM, Threadblock::kK, n_gld>, stride_a>;
    MemLoaderA LoaderA{A, As, total_threadId, blockIdx.y * Threadblock::kM * K,
                       K};
    using MemLoaderB = SharedLoaderMatrix<
        half, Threadblock::kK, Threadblock::kN + skew, n_gld, SpanTypeB,
        Index<Threadblock::kK, Threadblock::kN, n_gld>, stride_b>;
    MemLoaderB LoaderB{B, Bs, total_threadId, blockIdx.x * Threadblock::kN, N};

    const size_t warp_id = threadIdx.x / warpSize;

    const size_t warp_x = warp_id % warpCols;
    const size_t warp_y = warp_id / (num_warps / warpRows);
    const size_t AsWarpOffset = warp_y * Warp::kM;
    const size_t BsWarpOffset = warp_x * Warp::kN;

    constexpr size_t threadRows = Warp::kM / WMMAblock::kM;
    constexpr size_t threadCols = Warp::kN / WMMAblock::kN;

    C = &C[blockIdx.y * Threadblock::kM * N + blockIdx.x * Threadblock::kN +
           warp_y * Warp::kM * N + warp_x * Warp::kN];

    using RegisterTypeA =
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMAblock::kM,
                               WMMAblock::kN, WMMAblock::kK, half,
                               nvcuda::wmma::row_major>;
    RegisterTypeA A_frag[threadRows];
    using RegisterTypeB =
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMAblock::kM,
                               WMMAblock::kN, WMMAblock::kK, half,
                               nvcuda::wmma::row_major>;
    RegisterTypeB B_frag[threadCols];
    using RegisterTypeC =
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMAblock::kM,
                               WMMAblock::kN, WMMAblock::kK, half>;
    RegisterTypeC C_frag[threadRows][threadCols];
    for (size_t i = 0; i < threadRows; ++i) {
        for (size_t j = 0; j < threadCols; ++j) {
            nvcuda::wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }
    RegisterLoader<RegisterTypeA, MemLoaderA::cols_, WMMAblock::kM * As.cols_,
                   threadRows, 1>
        RegisterLoaderA(&As(AsWarpOffset, 0), A_frag);
    RegisterLoader<RegisterTypeB, MemLoaderB::cols_, WMMAblock::kN, threadCols,
                   1>
        RegisterLoaderB(&Bs(0, BsWarpOffset), B_frag);
    MatMul<RegisterTypeA, RegisterTypeB, RegisterTypeC, threadRows, threadCols,
           1>
        matmul(A_frag, B_frag, C_frag);

    for (size_t block = 0; block < K; block += Threadblock::kK) {
        LoaderA.load_blocking();
        LoaderB.load_blocking();
        LoaderA.next(Threadblock::kK);
        LoaderB.next(Threadblock::kK * N);
        __syncthreads();
        constexpr size_t wmma_steps = Threadblock::kK / WMMAblock::kK;
#pragma unroll
        for (size_t wmma_step = 0; wmma_step < wmma_steps; wmma_step++) {
            RegisterLoaderA.load();
            RegisterLoaderB.load();
            RegisterLoaderA.step(WMMAblock::kK);
            RegisterLoaderB.step(SpanTypeB::cols_ * WMMAblock::kN);
            matmul.compute();
        }
        RegisterLoaderA.reset(0);
        RegisterLoaderB.reset(0);
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

void BasicGEMM(half *A, half *B, half *C, size_t M, size_t N,
                                 size_t K, float alpha, float beta) {
    using tb = Block<256, 128, 32>;
    // Cutlass has often 256, 128, 64.... for sm80
    using wb = Block<64, 64, 32>;

    using wmma_block = Block<16, 16, 16>;
    assert(M % tb::kM == 0);
    assert(N % tb::kN == 0);
    static_assert(tb::kM % wb::kM == 0);
    static_assert(tb::kN % wb::kN == 0);
    dim3 gdim(N / tb::kN, M / tb::kM, 1);
    dim3 bdim(32 * (tb::kM / wb::kM) * (tb::kN / wb::kN), 1, 1);
    constexpr size_t skew = 8;
    constexpr size_t memsz =
        tb::kM * (tb::kK + skew) + tb::kK * (tb::kN + skew);
    constexpr size_t nbytes = memsz * sizeof(half);
    // std::cout << "Requesting: " << memsz * sizeof(half) << std::endl;
    BasicGEMM<tb, wb, wmma_block><<<gdim, bdim, nbytes>>>(A, B, C, M, N, K);
}
