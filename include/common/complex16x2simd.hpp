//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a SIMD implementation of the double precision complex type.
// The API is designed to (almost entirely) mirror that of the C++ standard library
// double precision complex type.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

namespace Qrack {

/** SIMD implementation of the double precision complex vector type of 2 complex numbers, only for AVX Apply2x2. */
struct Complex16x2Simd {
    __m256d _val2;

    inline Complex16x2Simd(){};
    inline Complex16x2Simd(const __m256d& v2) { _val2 = v2; }
    inline Complex16x2Simd(const double& r1, const double& i1, const double& r2, const double& i2)
    {
        _val2 = _mm256_set_pd(i1, r1, i2, r2);
    }
    inline Complex16x2Simd operator+(const Complex16x2Simd& other) const { return _mm256_add_pd(_val2, other._val2); }
    inline Complex16x2Simd operator+=(const Complex16x2Simd& other)
    {
        _val2 = _mm256_add_pd(_val2, other._val2);
        return _val2;
    }
    inline Complex16x2Simd operator-(const Complex16x2Simd& other) const { return _mm256_sub_pd(_val2, other._val2); }
    inline Complex16x2Simd operator-=(const Complex16x2Simd& other)
    {
        _val2 = _mm256_sub_pd(_val2, other._val2);
        return _val2;
    }
    inline Complex16x2Simd operator*(const Complex16x2Simd& other) const
    {
        return _mm256_add_pd(
            _mm256_mul_pd(_mm256_shuffle_pd(_val2, _val2, 5), _mm256_shuffle_pd((-other._val2), other._val2, 15)),
            _mm256_mul_pd(_val2, _mm256_shuffle_pd(other._val2, other._val2, 0)));
    }
    inline Complex16x2Simd operator*=(const Complex16x2Simd& other)
    {
        _val2 = _mm256_add_pd(
            _mm256_mul_pd(_mm256_shuffle_pd(_val2, _val2, 5), _mm256_shuffle_pd((-other._val2), other._val2, 15)),
            _mm256_mul_pd(_val2, _mm256_shuffle_pd(other._val2, other._val2, 0)));
        return _val2;
    }
    inline Complex16x2Simd operator*(const double rhs) const { return _mm256_mul_pd(_val2, _mm256_set1_pd(rhs)); }
    inline Complex16x2Simd operator-() const { return -_val2; }
    inline Complex16x2Simd operator*=(const double& other)
    {
        _val2 = _mm256_mul_pd(_val2, _mm256_set1_pd(other));
        return _val2;
    }
};

inline Complex16x2Simd dupeLo(const Complex16x2Simd& cmplx2)
{
    return _mm256_permute2f128_pd(cmplx2._val2, cmplx2._val2, 0);
}
inline Complex16x2Simd dupeHi(const Complex16x2Simd& cmplx2)
{
    return _mm256_permute2f128_pd(cmplx2._val2, cmplx2._val2, 17);
}
inline Complex16x2Simd matrixMul(
    const Complex16x2Simd& mtrxCol1, const Complex16x2Simd& mtrxCol2, const Complex16x2Simd& qubit)
{
    __m256d dupeLo = _mm256_permute2f128_pd(qubit._val2, qubit._val2, 0);
    __m256d dupeHi = _mm256_permute2f128_pd(qubit._val2, qubit._val2, 17);
    return _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(mtrxCol1._val2, mtrxCol1._val2, 5), _mm256_shuffle_pd(-dupeLo, dupeLo, 15)), _mm256_mul_pd(mtrxCol1._val2, _mm256_shuffle_pd(dupeLo, dupeLo, 0))), _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(mtrxCol2._val2, mtrxCol2._val2, 5), _mm256_shuffle_pd(-dupeHi, dupeHi, 15)), _mm256_mul_pd(mtrxCol2._val2, _mm256_shuffle_pd(dupeHi, dupeHi, 0))));
}
inline Complex16x2Simd matrixMul(
    const double& nrm, const Complex16x2Simd& mtrxCol1, const Complex16x2Simd& mtrxCol2, const Complex16x2Simd& qubit)
{
    __m256d dupeLo = _mm256_permute2f128_pd(qubit._val2, qubit._val2, 0);
    __m256d dupeHi = _mm256_permute2f128_pd(qubit._val2, qubit._val2, 17);
    return _mm256_mul_pd(_mm256_set1_pd(nrm), _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(mtrxCol1._val2, mtrxCol1._val2, 5), _mm256_shuffle_pd(-dupeLo, dupeLo, 15)), _mm256_mul_pd(mtrxCol1._val2, _mm256_shuffle_pd(dupeLo, dupeLo, 0))), _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(mtrxCol2._val2, mtrxCol2._val2, 5), _mm256_shuffle_pd(-dupeHi, dupeHi, 15)), _mm256_mul_pd(mtrxCol2._val2, _mm256_shuffle_pd(dupeHi, dupeHi, 0)))));
}
inline Complex16x2Simd operator*(const double lhs, const Complex16x2Simd& rhs)
{
    return _mm256_mul_pd(_mm256_set1_pd(lhs), rhs._val2);
}

} // namespace Qrack
