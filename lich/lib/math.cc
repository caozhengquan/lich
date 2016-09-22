#include "lich/lib/math.h"
#include "cblas.h"
#include <glog/logging.h>

#include <random>
#include <cstring>
#include <math.h>

namespace lich {


template <>
void lich_axpy<float>(const int N, const float alpha,
               const float* X, float* Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);                 
}
template <>
void lich_axpy<double>(const int N, const double alpha,
               const double* X, double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);                 
}

template <>
float lich_dot(const int N, const float* X, const float* Y) {
  return cblas_sdot(N, X, 1, Y, 1);                 
}
template <>
double lich_dot(const int N, const double* X, const double* Y) {
  return cblas_ddot(N, X, 1, Y, 1);                 
}

template <>
float lich_strided_dot(const int N, const float* X, const int incx,
                       const float* Y, const int incy) {  
  return cblas_sdot(N, X, incx, Y, incy);
}
template <>
double lich_strided_dot(const int N, const double* X, const int incx,
                       const double* Y, const int incy) {  
  return cblas_ddot(N, X, incx, Y, incy);
}

template <>
void lich_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K,
               const float alpha, const float* A, const float* B,
               const float beta, float* C) {
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda,
              B, ldb, beta, C, N);                   
}
template <>
void lich_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K,
               const double alpha, const double* A, const double* B,
               const double beta, double* C) {
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda,
              B, ldb, beta, C, N);                   
}

template <>
void lich_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
               const float alpha, const float* A, const float* x,
               const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}
template <>
void lich_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
               const double alpha, const double* A, const double* x,
               const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <typename Dtype>
void lich_rng_gaussian(const int N, const Dtype mean, const Dtype std, 
                       Dtype* data) {
  if (data == nullptr) return;
  CHECK_GT(std, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<Dtype> norm(mean, std);
  for (int i = 0; i < N; ++i) {
    data[i] = norm(gen);
  }
}

// explicit instantiation for efficiency.(Only generate one lib code)
template void lich_rng_gaussian<float>(const int N, const float mean,
                                       const float std, float* data);
template void lich_rng_gaussian<double>(const int N, const double mean,
                                       const double std, double* data);                                       

template <typename Dtype>
void lich_rng_uniform(const int N, const Dtype min, const Dtype max, Dtype* data) {
  if (data == nullptr) return;
  CHECK_GE(N, 0);
  CHECK_LE(min, max);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Dtype> uniform(min, max);
  for (int i = 0; i < N; ++i) {
    data[i] = uniform(gen);
  }
}

template void lich_rng_uniform<float>(const int N, const float min, const float max, float* data);
template void lich_rng_uniform<double>(const int N, const double min, const double max, double* data);

template <typename Dtype>
void lich_set(const int N, const Dtype value, Dtype* data) {
  if (data == nullptr) return;
  if (value == 0) {
    memset(data, 0, sizeof(Dtype) * N);
    return;
  }
  for (int i = 0; i < N; ++i) {
    data[i] = value;
  }
}

template void lich_set<int>(const int N, const int value, int* data);
template void lich_set<float>(const int N, const float value, float* data);
template void lich_set<double>(const int N, const double value, double* data);

template <>
void lich_scal(const int N, const float alpha, float* X) {
  cblas_sscal(N, alpha, X, 1);
}
template <>
void lich_scal(const int N, const double alpha, double* X) {
  cblas_dscal(N, alpha, X, 1);
}

template <typename Dtype>
void lich_sign(const int N, const Dtype* X, Dtype* Y) {
  for (int i = 0; i < N; ++i) {
    if (X[i] > 0) Y[i] = 1;
    else Y[i] = -1;
  }
}

template void lich_sign<int>(const int N, const int* X, int* Y);
template void lich_sign<float>(const int N, const float* X, float* Y);
template void lich_sign<double>(const int N, const double* X, double* Y);

template <typename Dtype>
void lich_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X == Y) return;
  std::memcpy(Y, X, N * sizeof(Dtype));
}

template void lich_copy<int>(const int N, const int* X, int* Y);
template void lich_copy<float>(const int N, const float* X, float* Y);
template void lich_copy<double>(const int N, const double* X, double* Y);

template <typename Dtype>
void lich_exp(const int N, const Dtype* X, Dtype* Y) {
  //vsExp(N, X, Y) not available for opencblas
  for (int i = 0; i < N; ++i) {
    Y[i] = std::exp(X[i]);
  }
}

template void lich_exp<int>(const int N, const int* X, int* Y);
template void lich_exp<float>(const int N, const float* X, float* Y);
template void lich_exp<double>(const int N, const double* X, double* Y);

template <typename Dtype>
void lich_div(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = a[i] / b[i];
  }
}

template void lich_div<int>(const int N, const int* a, const int* b, int* y);
template void lich_div<float>(const int N, const float* a, const float* b, float* y);
template void lich_div<double>(const int N, const double* a, const double* b, double* y);

template <typename Dtype>
void lich_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = a[i] * b[i];
  }
}

template void lich_mul<int>(const int N, const int* a, const int* b, int* y);
template void lich_mul<float>(const int N, const float* a, const float* b, float* y);
template void lich_mul<double>(const int N, const double* a, const double* b, double* y);

template <typename Dtype>
void lich_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = a[i] - b[i];
  }
}

template void lich_sub<int>(const int N, const int* a, const int* b, int* y);
template void lich_sub<float>(const int N, const float* a, const float* b, float* y);
template void lich_sub<double>(const int N, const double* a, const double* b, double* y);

} // namespace lich 