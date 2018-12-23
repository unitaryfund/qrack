//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a SIMD implementation of the double precision complex type.
// The API is designed to (almost entirely) mirror that of the C++ standard library
// double precision complex type.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <cmath>

#include <emmintrin.h>

#if ENABLE_AVX
#include <smmintrin.h>
#endif

namespace Qrack {

/** SIMD implementation of the double precision complex type. */
struct Complex16Simd {
    __m128d _val;

    inline Complex16Simd(){};
    inline Complex16Simd(const __m128d& v) { _val = v; }
    inline Complex16Simd(const double& real, const double& imag) { _val = _mm_set_pd(imag, real); }
    inline Complex16Simd operator+(const Complex16Simd& other) const { return _mm_add_pd(_val, other._val); }
    inline Complex16Simd operator+=(const Complex16Simd& other)
    {
        _val = _mm_add_pd(_val, other._val);
        return _val;
    }
    inline Complex16Simd operator-(const Complex16Simd& other) const { return _mm_sub_pd(_val, other._val); }
    inline Complex16Simd operator-=(const Complex16Simd& other)
    {
        _val = _mm_sub_pd(_val, other._val);
        return _val;
    }
    inline Complex16Simd operator*(const Complex16Simd& other) const
    {
        return _mm_add_pd(_mm_mul_pd(_mm_shuffle_pd(_val, _val, 1), _mm_shuffle_pd((-other._val), other._val, 3)),
            _mm_mul_pd(_val, _mm_shuffle_pd(other._val, other._val, 0)));
    }
    inline Complex16Simd operator*=(const Complex16Simd& other)
    {
        _val = _mm_add_pd(_mm_mul_pd(_mm_shuffle_pd(_val, _val, 1), _mm_shuffle_pd((-other._val), other._val, 3)),
            _mm_mul_pd(_val, _mm_shuffle_pd(other._val, other._val, 0)));
        return _val;
    }
    inline Complex16Simd operator*(const double rhs) const { return _mm_mul_pd(_val, _mm_set1_pd(rhs)); }
    inline Complex16Simd operator/(const Complex16Simd& other) const
    {
#if ENABLE_AVX
        double nrm = _mm_cvtsd_f64(_mm_dp_pd(other._val, other._val, 0xf1));
#else
        __v2df temp = (__v2df)_mm_mul_pd(other._val, other._val);
        double nrm = (temp[0] + temp[1]);
#endif
        return _mm_div_pd(
            _mm_add_pd(_mm_mul_pd(_mm_shuffle_pd(_val, _val, 1), _mm_shuffle_pd(other._val, -(other._val), 3)),
                _mm_mul_pd(_val, _mm_shuffle_pd(other._val, other._val, 0))),
            _mm_set1_pd(nrm));
    }
    inline Complex16Simd operator/=(const Complex16Simd& other)
    {
#if ENABLE_AVX
        double nrm = _mm_cvtsd_f64(_mm_dp_pd(other._val, other._val, 0xf1));
#else
        __v2df temp = (__v2df)_mm_mul_pd(other._val, other._val);
        double nrm = (temp[0] + temp[1]);
#endif
        _val = _mm_div_pd(
            _mm_add_pd(_mm_mul_pd(_mm_shuffle_pd(_val, _val, 1), _mm_shuffle_pd(other._val, -(other._val), 3)),
                _mm_mul_pd(_val, _mm_shuffle_pd(other._val, other._val, 0))),
            _mm_set1_pd(nrm));
        return _val;
    }
    inline Complex16Simd operator/(const double rhs) const { return _mm_div_pd(_val, _mm_set1_pd(rhs)); }
    inline Complex16Simd operator/=(const double rhs)
    {
        _val = _mm_div_pd(_val, _mm_set1_pd(rhs));
        return _val;
    }
    inline Complex16Simd operator-() const { return -_val; }
    inline Complex16Simd operator*=(const double& other)
    {
        _val = _mm_mul_pd(_val, _mm_set1_pd(other));
        return _val;
    }
    inline bool operator==(const Complex16Simd& rhs) const
    {
        __v2df vec = (__v2df)(_val == rhs._val);
        return vec[0] && vec[1];
    }
    inline bool operator!=(const Complex16Simd& rhs) const
    {
        __v2df vec = (__v2df)(_val != rhs._val);
        return vec[0] && vec[1];
    }
};

inline Complex16Simd operator*(const double& lhs, const Complex16Simd& rhs)
{
    return _mm_mul_pd(_mm_set1_pd(lhs), rhs._val);
}
inline Complex16Simd operator/(const double& lhs, const Complex16Simd& rhs)
{
    __v2df temp = (__v2df)_mm_mul_pd(rhs._val, rhs._val);
    double denom = temp[0] + temp[1];
    temp = (__v2df)_mm_div_pd(_mm_mul_pd(rhs._val, _mm_set1_pd(lhs)), _mm_set1_pd(denom));
    return Complex16Simd(temp[0], -temp[1]);
}

inline double real(const Complex16Simd& cmplx) { return ((__v2df)(cmplx._val))[0]; }
inline double imag(const Complex16Simd& cmplx) { return ((__v2df)(cmplx._val))[1]; }

inline double arg(const Complex16Simd& cmplx)
{
    if (cmplx == Complex16Simd(0.0, 0.0))
        return 0.0;
    return atan2(imag(cmplx), real(cmplx));
}
inline Complex16Simd conj(const Complex16Simd& cmplx)
{
    return _mm_shuffle_pd(cmplx._val, _mm_sub_pd(_mm_set1_pd(0.0), cmplx._val), 2);
}
inline double norm(const Complex16Simd& cmplx)
{
#if ENABLE_AVX
    return _mm_cvtsd_f64(_mm_dp_pd(cmplx._val, cmplx._val, 0xf1));
#else
    __v2df temp = (__v2df)_mm_mul_pd(cmplx._val, cmplx._val);
    return (temp[0] + temp[1]);
#endif
}
inline double abs(const Complex16Simd& cmplx) { return sqrt(norm(cmplx)); }
inline Complex16Simd polar(const double rho, const double theta = 0)
{
    return _mm_set1_pd(rho) * _mm_set_pd(sin(theta), cos(theta));
}
inline Complex16Simd sqrt(const Complex16Simd& cmplx)
{
    double theta = arg(cmplx);
    return std::sqrt(abs(cmplx)) * Complex16Simd(std::cos(theta / 2.0), std::sin(theta / 2.0));
}
} // namespace Qrack
