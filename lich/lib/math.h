#ifndef LICH_LIB_MATH_H_
#define LICH_LIB_MATH_H_

#include "cblas.h"

namespace lich {

// Y = alpha * X + Y
template <typename Dtype>
void lich_axpy(const int N, const Dtype alpha, 
               const Dtype* X, Dtype* Y);
     
// Y = alpha * X + Y
template <typename Dtype>
Dtype lich_dot(const int N, const Dtype* X, const Dtype* Y);

// Y = alpha * X + Y strided
template <typename Dtype>
Dtype lich_strided_dot(const int N, const Dtype* X, const int incx,
                       const Dtype* Y, const int incy);             

// C = alpha * A * B + beta * C; 
// A: M * K, B: K * N, C: M * N
template <typename Dtype>
void lich_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K,
               const Dtype alpha, const Dtype* A, const Dtype* B,
               const Dtype beta, Dtype* C);

// y = alpha * A * x + beta * y;
// A: M * N, x: M * 1, y: M * 1
template <typename Dtype>
void lich_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
               const Dtype alpha, const Dtype* A, const Dtype* x,
               const Dtype beta, Dtype* y);

// Set data to Guass(mean, std)
template <typename Dtype>
void lich_rng_gaussian(const int N, const Dtype mean, const Dtype std, Dtype* data);

// Set data to Uniform(min, max)
template <typename Dtype>
void lich_rng_uniform(const int N, const Dtype min, const Dtype max, Dtype* data);

// Set data to value
template <typename Dtype>
void lich_set(const int N, const Dtype value, Dtype* data);

// X = alpha * X
template <typename Dtype>
void lich_scal(const int N, const Dtype alpha, Dtype* X);

// Y = sign(X)
template <typename Dtype>
void lich_sign(const int N, const Dtype* X, Dtype* Y);

// Y = X
template <typename Dtype>
void lich_copy(const int N, const Dtype* X, Dtype* Y);

// Elementwise exp 
template <typename Dtype>
void lich_exp(const int N, const Dtype* X, Dtype* Y);

// Elementwise divide
template <typename Dtype>
void lich_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

// Elementwise multiply
template <typename Dtype>
void lich_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

// Elementwise substract
template <typename Dtype>
void lich_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

} // namespace lich 

#endif