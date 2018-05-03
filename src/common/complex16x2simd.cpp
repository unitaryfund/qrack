//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a SIMD implementation of the double precision complex type.
// The API is designed to (almost entirely) mirror that of the C++ standard library
// double precision complex type.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#define CMPLX_INC
#include "common/complex16x2simd.hpp"

namespace Qrack {

/** SIMD implementation of the double precision complex vector type of length 2. */

Complex16x2Simd::Complex16x2Simd() {}
Complex16x2Simd::Complex16x2Simd(__m256d v2) { _val2 = v2; }
Complex16x2Simd::Complex16x2Simd(double real1, double imag1, double real2, double imag2) { _val2 = _mm256_set_pd(imag1, real1, imag2, real2); }
Complex16x2Simd Complex16x2Simd::operator+(const Complex16x2Simd& other) const
{
    return Complex16x2Simd(_mm256_add_pd(_val2, other._val2));
}
Complex16x2Simd Complex16x2Simd::operator+=(const Complex16x2Simd& other)
{
    _val2 = _mm256_add_pd(_val2, other._val2);
    return Complex16x2Simd(_val2);
}
Complex16x2Simd Complex16x2Simd::operator-(const Complex16x2Simd& other) const
{
    return Complex16x2Simd(_mm256_sub_pd(_val2, other._val2));
}
Complex16x2Simd Complex16x2Simd::operator-=(const Complex16x2Simd& other)
{
    _val2 = _mm256_sub_pd(_val2, other._val2);
    return Complex16x2Simd(_val2);
}
Complex16x2Simd Complex16x2Simd::operator*(const Complex16x2Simd& other) const
{
    return Complex16x2Simd(_mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(_val2, _val2, 5), _mm256_shuffle_pd((-other._val2), other._val2, 15)),
        _mm256_mul_pd(_val2, _mm256_shuffle_pd(other._val2, other._val2, 0))));
}
Complex16x2Simd Complex16x2Simd::operator*=(const Complex16x2Simd& other)
{
    _val2 = _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(_val2, _val2, 5), _mm256_shuffle_pd((-other._val2), other._val2, 15)),
        _mm256_mul_pd(_val2, _mm256_shuffle_pd(other._val2, other._val2, 0)));
    return Complex16x2Simd(_val2);
}
Complex16x2Simd Complex16x2Simd::operator*(const double rhs) const { return _mm256_mul_pd(_val2, _mm256_set1_pd(rhs)); }
Complex16x2Simd Complex16x2Simd::operator-() const { return -1.0 * _val2; }
Complex16x2Simd Complex16x2Simd::operator*=(const double& other)
{
    _val2 = _mm256_mul_pd(_val2, _mm256_set1_pd(other));
    return Complex16x2Simd(_val2);
}

Complex16x2Simd dupeLo(const Complex16x2Simd& cmplx2) { return _mm256_permute2f128_pd(cmplx2._val2, cmplx2._val2, 0); }
Complex16x2Simd dupeHi(const Complex16x2Simd& cmplx2) { return _mm256_permute2f128_pd(cmplx2._val2, cmplx2._val2, 17); }
// Imperative function definitions:
Complex16x2Simd operator*(const double lhs, const Complex16x2Simd& rhs) { return _mm256_mul_pd(_mm256_set1_pd(lhs), rhs._val2); }
double norm(const Complex16x2Simd& cmplx2)
{
    __v4df temp = (__v4df)_mm256_mul_pd(cmplx2._val2, cmplx2._val2);
    return (temp[0] + temp[1] + temp[2] + temp[3]);
}
} // namespace Qrack
