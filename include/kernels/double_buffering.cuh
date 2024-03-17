#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <sys/cdefs.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "load_shared.cuh"
#include "utils.cuh"

template <class Threadblock, class Warp, class WMMAblock>
__global__ void DoubleBuffering(const half *__restrict__ A,
                                const half *__restrict__ B, half *C, size_t N,
                                size_t M, size_t K) {
    assert(M % Threadblock::kM == 0);
    assert(N % Threadblock::kN == 0);
    assert(K % Threadblock::kK == 0);

    static_assert(Threadblock::kM >= Warp::kM);
    static_assert(Threadblock::kN >= Warp::kN);
    static_assert(Threadblock::kK >= Warp::kK);

    constexpr size_t warpRows = Threadblock::kM / Warp::kM;
    constexpr size_t warpCols = Threadblock::kN / Warp::kN;
    constexpr size_t num_warps = warpRows * warpCols;

    constexpr size_t skew = 8;

    extern __shared__ half shmem[];
    using SpanTypeA = Span<half, 2 * Threadblock::kM, Threadblock::kK + skew>;
    SpanTypeA As(&shmem[0]);
    using SpanTypeB = Span<half, 2 * Threadblock::kK, Threadblock::kN + skew>;
    SpanTypeB Bs(As.end());

    const size_t total_threadId = threadIdx.x;
    constexpr size_t thread_num = num_warps * 32;
    constexpr size_t CUDA_VECTORIZED_BITS_LOAD = 128;
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

    using RegisterTypeA =
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMAblock::kM,
                               WMMAblock::kN, WMMAblock::kK, half,
                               nvcuda::wmma::row_major>;
    RegisterTypeA A_frag[2 * threadRows];
    using RegisterTypeB =
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMAblock::kM,
                               WMMAblock::kN, WMMAblock::kK, half,
                               nvcuda::wmma::row_major>;
    RegisterTypeB B_frag[2 * threadCols];
    using RegisterTypeC =
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMAblock::kM,
                               WMMAblock::kN, WMMAblock::kK, half>;
    RegisterTypeC C_frag[threadRows][threadCols];
    for (size_t i = 0; i < threadRows; ++i) {
        for (size_t j = 0; j < threadCols; ++j) {
            nvcuda::wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }
    RegisterLoader<RegisterTypeA, LoaderA.cols_, WMMAblock::kM * As.cols_,
                   threadRows, 2>
        RegisterLoaderA(&As(AsWarpOffset, 0), A_frag);
    RegisterLoader<RegisterTypeB, LoaderB.cols_, WMMAblock::kN, threadCols, 2>
        RegisterLoaderB(&Bs(0, BsWarpOffset), B_frag);
    MatMul<RegisterTypeA, RegisterTypeB, RegisterTypeC, threadRows, threadCols,
           2>
        matmul(A_frag, B_frag, C_frag);

    LoaderA.load(0);
    LoaderB.load(0);
    LoaderA.next(Threadblock::kK);
    LoaderB.next(Threadblock::kK * N);
    CP_ASYNC_COMMIT_GROUP();
    LoaderA.load(1);
    LoaderB.load(1);
    LoaderA.next(Threadblock::kK);
    LoaderB.next(Threadblock::kK * N);
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(1);  // 1 = Wait until 1 recent async groups are pending
    __syncthreads();
    RegisterLoaderA.load(0);
    RegisterLoaderB.load(0);
    RegisterLoaderA.step(WMMAblock::kK);
    RegisterLoaderB.step(SpanTypeB::cols_ * WMMAblock::kN);
    size_t counter = 1;
    for (size_t block = 0; block < K - 2 * Threadblock::kK;
         block += Threadblock::kK) {
        constexpr size_t k_steps = Threadblock::kK / WMMAblock::kK;
#pragma unroll
        for (size_t i = 0; i < k_steps; ++i) {
            size_t current = i % 2;
            size_t next = (i + 1) % 2;
            RegisterLoaderA.load(next);
            RegisterLoaderB.load(next);
            RegisterLoaderA.step(WMMAblock::kK);
            RegisterLoaderB.step(SpanTypeB::cols_ * WMMAblock::kN);
            matmul.compute(current);
            if (i == 0) {
                LoaderA.load(counter ^ 1);
                LoaderB.load(counter ^ 1);
                LoaderA.next(Threadblock::kK);
                LoaderB.next(Threadblock::kK * N);
                CP_ASYNC_COMMIT_GROUP();
                CP_ASYNC_WAIT_GROUP(1);
                __syncthreads();
                RegisterLoaderA.reset(counter * MemLoaderA::size_);
                RegisterLoaderB.reset(counter * MemLoaderB::size_);
                counter ^= 1;
            }
        }
        __syncthreads();
    }
    RegisterLoaderA.load(1);
    RegisterLoaderB.load(1);
    RegisterLoaderA.step(WMMAblock::kK);
    RegisterLoaderB.step(SpanTypeB::cols_ * WMMAblock::kN);
    matmul.compute(0);
    RegisterLoaderA.reset(counter * MemLoaderA::size_);
    RegisterLoaderB.reset(counter * MemLoaderB::size_);
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
    RegisterLoaderA.load(0);
    RegisterLoaderB.load(0);
    RegisterLoaderA.step(WMMAblock::kK);
    RegisterLoaderB.step(SpanTypeB::cols_ * WMMAblock::kN);
    matmul.compute(1);
    RegisterLoaderA.load(1);
    RegisterLoaderB.load(1);
    matmul.compute(0);
    matmul.compute(1);
    __syncthreads();
    // C = &C[blockIdx.y * Threadblock::kM * N + blockIdx.x * Threadblock::kN +
    // warp_y * Warp::kM * N + warp_x * Warp::kN];
    //#pragma unroll
    // for (size_t i = 0; i < threadRows; ++i) {
    //#pragma unroll
    // for (size_t j = 0; j < threadCols; ++j) {
    // nvcuda::wmma::store_matrix_sync(
    //&C[i * WMMAblock::kM * N + j * WMMAblock::kK], C_frag[i][j], N,
    // nvcuda::wmma::mem_row_major);
    //}
    //}
    // Write data to C via shared memory. Given ~2TFLOPS.
    Span<half, Threadblock::kM, Threadblock::kN + skew> Cs(&shmem[0]);
#pragma unroll
    for (size_t i = 0; i < threadRows; ++i) {
#pragma unroll
        for (size_t j = 0; j < threadCols; ++j) {
            nvcuda::wmma::store_matrix_sync(
                &Cs(warp_y * Warp::kM + i * WMMAblock::kM,
                    warp_x * Warp::kN + j * WMMAblock::kN),
                C_frag[i][j], Threadblock::kN + skew,
                nvcuda::wmma::mem_row_major);
        }
    }
    const size_t thread_id = threadIdx.x % 32;
    C = &C[blockIdx.y * Threadblock::kM * N + blockIdx.x * Threadblock::kN +
           warp_y * Warp::kM * N + warp_x * Warp::kN];
    constexpr size_t threadsPerRow = Warp::kN / n_gld;
    constexpr size_t rowsFilled = 32 / threadsPerRow;
    constexpr size_t storeRows = Warp::kM / rowsFilled;
    const size_t store_x = thread_id % threadsPerRow;
    const size_t store_y = thread_id / threadsPerRow;
#pragma unroll
    for (size_t i = 0; i < storeRows; ++i) {
        size_t row = warp_y * Warp::kM + i * rowsFilled + store_y;
        // assert(warp_y * Warp::kM + i * rowsFilled + store_y <
        // Threadblock::kM); assert(warp_x * Warp::kN + n_gld * store_x <
        // Threadblock::kN);
        half *ptr_cs = &Cs(warp_y * Warp::kM + i * rowsFilled + store_y,
                           warp_x * Warp::kN + n_gld * store_x);
        int4 t = reinterpret_cast<const int4 *>(ptr_cs)[0];
        half *dst_ptr = &C[(i * rowsFilled + store_y) * N + n_gld * store_x];
        reinterpret_cast<int4 *>(dst_ptr)[0] = t;
    }
}

// template <typename T, typename accum_type>
void DoubleBuffering(half *A, half *B, half *C, size_t M, size_t N, size_t K,
                     float alpha, float beta) {
    using tb = Block<256, 128, 32>;
    // using tb = Block<128, 128, 32>;
    //  Cutlass has often 256, 128, 64.... for sm80
    // using wb = Block<32, 32, 32>;
    using wb = Block<128, 64, 32>;

    using wmma_block = Block<16, 16, 16>;
    assert(M % tb::kM == 0);
    assert(N % tb::kN == 0);
    static_assert(tb::kM % wb::kM == 0);
    static_assert(tb::kN % wb::kN == 0);
    dim3 gdim(N / tb::kN, M / tb::kM, 1);  // x values for cols, y for rows
    dim3 bdim(32 * (tb::kM / wb::kM) * (tb::kN / wb::kN), 1, 1);
    constexpr size_t skew = 8;
    constexpr size_t sz_buffers_a_b =
        ((2 * tb::kM * (tb::kK + skew)) + (2 * tb::kK * (tb::kN + skew)));
    constexpr size_t sz_buffer_c = tb::kM * (tb::kN + skew);
    constexpr size_t nbytes =
        std::max<size_t>(sz_buffers_a_b, sz_buffer_c) * sizeof(half);
    // std::cout << "Requsting " << nbytes << std::endl;
    cudaFuncSetAttribute(DoubleBuffering<tb, wb, wmma_block>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes);
    DoubleBuffering<tb, wb, wmma_block>
        <<<gdim, bdim, nbytes>>>(A, B, C, M, N, K);
}
