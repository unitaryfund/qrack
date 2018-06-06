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

#include <xmmintrin.h>
#include <pmmintrin.h>

namespace Qrack {
    
union C8ToF4 {
    __m128d complex;
    float floats[4];
    C8ToF4(_m128d cmplx) {
        complex = cmplx;
    }
};

/** SIMD implementation of the double precision complex type. */
struct Complex8Simd {
    __m128d _val;

    inline Complex8Simd(){};
    inline Complex8Simd(const __m128d& v) { _val = v; }
    inline Complex8Simd(const double& real, const double& imag) { _val = _mm_set_ps(imag, real, 0, 0); }
    inline Complex8Simd operator+(const Complex8Simd& other) const { return _mm_add_ps(_val, other._val); }
    inline Complex8Simd operator+=(const Complex8Simd& other)
    {
        _val = _mm_add_ps(_val, other._val);
        return _val;
    }
    inline Complex8Simd operator-(const Complex8Simd& other) const { return _mm_sub_ps(_val, other._val); }
    inline Complex8Simd operator-=(const Complex8Simd& other)
    {
        _val = _mm_sub_ps(_val, other._val);
        return _val;
    }
    inline Complex8Simd operator*(const Complex8Simd& other) const
    {
        return _mm_addsub_ps(_mm_mul_ps(_mm_shuffle_ps(_val, _val, 235), _mm_shuffle_ps(other._val, other._val, 250), _mm_mul_ps(_mm_shuffle_ps(_val, _val, 235), _mm_shuffle_ps(other._val, other._val, 175));
    }
    inline Complex8Simd operator*=(const Complex8Simd& other)
    {
        _val = _mm_addsub_ps(_mm_mul_ps(_mm_shuffle_ps(_val, _val, 235), _mm_shuffle_ps(other._val, other._val, 250), _mm_mul_ps(_mm_shuffle_ps(_val, _val, 235), _mm_shuffle_ps(other._val, other._val, 175));
        return _val;
    }
    inline Complex8Simd operator*(const double rhs) const { return _mm_mul_ps(_val, _mm_set1_ps(rhs)); }
    inline Complex8Simd operator/(const Complex8Simd& other) const
    {
        C8ToF4 temp(_mm_mul_ps(other._val, other._val));
        float nrm = (temp.floats[2] + temp.floats[3]);

        return _mm_div_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(_val, _val, 235), _mm_shuffle_ps(other._val, other._val, 250), _mm_mul_ps(_mm_shuffle_ps(_val, -_val, 235), _mm_shuffle_ps(other._val, other._val, 175)),_mm_set1_pd(nrm));
    }
    inline Complex8Simd operator/=(const Complex8Simd& other)
    {
        C8ToF4 temp(temp.complex = _mm_mul_ps(other._val, other._val));
        float nrm = (temp.floats[2] + temp.floats[3]);

        _val = _mm_div_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(_val, _val, 235), _mm_shuffle_ps(other._val, other._val, 250), _mm_mul_ps(_mm_shuffle_ps(_val, -_val, 235), _mm_shuffle_ps(other._val, other._val, 175)),_mm_set1_pd(nrm));
        return _val;
    }
    inline Complex8Simd operator/(const double rhs) const { return _mm_div_ps(_val, _mm_set1_ps(rhs)); }
    inline Complex8Simd operator/=(const double rhs)
    {
        _val = _mm_div_ps(_val, _mm_set1_ps(rhs));
        return _val;
    }
    inline Complex8Simd operator-() const { return -_val; }
    inline Complex8Simd operator*=(const double& other)
    {
        _val = _mm_mul_ps(_val, _mm_set1_ps(other));
        return _val;
    }
    inline bool operator==(const Complex8Simd& rhs) const
    {
        C8ToF4 vec(_val == rhs._val);
        return vec[2] && vec[3];
    }
    inline bool operator!=(const Complex8Simd& rhs) const
    {
        C8ToF4 vec(_val != rhs._val);
        return vec[2] && vec[3];
    }
};

inline Complex8Simd operator*(const double& lhs, const Complex8Simd& rhs)
{
    return _mm_mul_ps(_mm_set1_ps(lhs), rhs._val);
}
inline Complex8Simd operator/(const double& lhs, const Complex8Simd& rhs)
{
    C8ToF4 temp(_mm_mul_pd(rhs._val, rhs._val));
    double denom = temp[2] + temp[3];
    temp.complex = _mm_div_pd(_mm_mul_pd(rhs._val, _mm_set1_pd(lhs)), _mm_set1_pd(denom));
    return Complex8Simd(temp[2], -temp[3]);
}

inline double real(const Complex8Simd& cmplx) { return ((cmplx._val))[0]; }
inline double imag(const Complex8Simd& cmplx) { return ((__v2df)(cmplx._val))[1]; }

inline double arg(const Complex8Simd& cmplx)
{
    if (cmplx == Complex8Simd(0.0, 0.0))
        return 0.0;
    return atan2(imag(cmplx), real(cmplx));
}
inline Complex8Simd conj(const Complex8Simd& cmplx)
{
    return _mm_shuffle_ps(cmplx._val, _mm_sub_pd(_mm_set1_pd(0.0), cmplx._val), 2);
}
inline double norm(const Complex8Simd& cmplx)
{
    C8ToF4 temp(_mm_mul_pd(cmplx._val, cmplx._val));
    return (temp[2] + temp[3]);
}
inline double abs(const Complex8Simd& cmplx) { return sqrt(norm(cmplx)); }
inline Complex8Simd polar(const double rho, const double theta = 0)
{
    return _mm_set1_ps(rho) * _mm_set_ps(sin(theta), cos(theta));
}
} // namespace Qrack
