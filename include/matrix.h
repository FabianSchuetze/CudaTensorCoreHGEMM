// A simple matrix class for host and device memory, similar to a thrust vector
//
#pragma once
#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <stdexcept>

#include "utils.cuh"

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

template <typename T>
class Matrix {
   public:
    Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), host(new T[rows * cols * sizeof(T)]) {
        DeviceAlloc();
    };
    ~Matrix() {
        delete[] host;
        cudaFree(device);
    };
    // Delete copy constructor /assignment operator because the Matrix has
    // reference semantics and still a destructor that cleans memory
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    void sync(cudaMemcpyKind kind) {
        cudaError_t status{};
        if (kind == cudaMemcpyHostToDevice) {
            status = cudaMemcpy(device, host, n_ele() * sizeof(T), kind);
        } else if (kind == cudaMemcpyDeviceToHost) {
            status = cudaMemcpy(host, device, n_ele() * sizeof(T), kind);
        } else {
            throw std::runtime_error{"Not supported"};
        };
        cudaCheck(status);
    }
    T* device_ptr() { return device; };
    T* host_ptr() { return host; }
    const T * host_ptr() const { return host; }
    T& operator[](size_t i) { return *(host + i); };
    T operator[](size_t i) const { return *(host + i); };
    [[nodiscard]] size_t n_ele() const noexcept { return rows_ * cols_; };

   private:
    size_t rows_;
    size_t cols_;
    T* device;
    T* host;
    void DeviceAlloc() {
        cudaMalloc((void**)&device, rows_ * cols_ * sizeof(T));
    };
};

template <typename T>
void random_fill_and_sync(Matrix<T>& matrix) {
    std::default_random_engine engine{0};
    std::uniform_real_distribution<float> uniform(-1, 1);
    for (size_t i = 0; i < matrix.n_ele(); ++i) {
        matrix[i] = (T)(uniform(engine));
    }
    matrix.sync(cudaMemcpyHostToDevice);
}

template <typename T>
void fill_and_sync(Matrix<T>& matrix, T val) {
    for (size_t i = 0; i < matrix.n_ele(); ++i) {
        matrix[i] = val;
    }
    matrix.sync(cudaMemcpyHostToDevice);
};

template <typename T>
void fill_iota_and_sync(Matrix<T>& matrix) {
    for (unsigned int i = 0; i < matrix.n_ele(); ++i) {
        matrix[i] = static_cast<T>(i);
    }
    matrix.sync(cudaMemcpyHostToDevice);
};

template <typename T>
bool compareMatrices(const Matrix<T>& A, const Matrix<T>& B) {
    assert(A.n_ele() == B.n_ele());
    double tot_diff{0}, tot_b{0};
    for (int i = 0; i < A.n_ele(); ++i) {
        float a = (float)A[i];
        float b = (float)B[i];
        float diff = std::abs(a - b);
        tot_diff += (double)diff;
        tot_b += (double)std::abs(b);
        if (diff > 1.0) {  // great than 2 %
            std::cout << "at " << i << " the difference is: " << diff
                      << ", with " << a << " vs " << b << std::endl;
            return false;
        }
    }
    int i = 0;
    //std::cout << "Tot diff: tot b " << tot_diff << ", " << tot_b << std::endl;
    //std::cout << "The same, average error: " << tot_diff / tot_b << "; at " << i
              //<< ", with " << (float)A[i] << " vs " << (float)B[i] << std::endl;
    return true;
}

