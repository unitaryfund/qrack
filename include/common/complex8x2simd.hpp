//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a SIMD implementation of the float precision complex type.
// The API is designed to (almost entirely) mirror that of the C++ standard library
// float precision complex type.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#if defined(_WIN32)
#include <intrin.h>
#else
#include <xmmintrin.h>
#endif

#include <complex>

namespace Qrack {

static const __m128 SIGNMASK = _mm_set_ps(0.0f, -0.0f, 0.0f, -0.0f);

/** SIMD implementation of the float precision complex vector type of 2 complex numbers, only for COMPLEX_X_2 Apply2x2.
 */
struct Complex8x2Simd {
    __m128 _val2;

    inline Complex8x2Simd(){};
    inline Complex8x2Simd(const __m128& v2) { _val2 = v2; }
    inline Complex8x2Simd(const float& r1, const float& i1, const float& r2, const float& i2)
    {
        _val2 = _mm_set_ps(i2, r2, i1, r1);
    }
    inline Complex8x2Simd operator+(const Complex8x2Simd& other) const { return _mm_add_ps(_val2, other._val2); }
    inline Complex8x2Simd operator+=(const Complex8x2Simd& other)
    {
        _val2 = _mm_add_ps(_val2, other._val2);
        return _val2;
    }
    inline Complex8x2Simd operator-(const Complex8x2Simd& other) const { return _mm_sub_ps(_val2, other._val2); }
    inline Complex8x2Simd operator-=(const Complex8x2Simd& other)
    {
        _val2 = _mm_sub_ps(_val2, other._val2);
        return _val2;
    }
    inline Complex8x2Simd operator*(const Complex8x2Simd& other) const
    {
        const __m128& oVal2 = other._val2;
        return _mm_add_ps(
            _mm_mul_ps(_mm_shuffle_ps(_val2, _val2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(oVal2, oVal2, 245))),
            _mm_mul_ps(_val2, _mm_shuffle_ps(oVal2, oVal2, 160)));
    }
    inline Complex8x2Simd operator*=(const Complex8x2Simd& other)
    {
        const __m128& oVal2 = other._val2;
        _val2 = _mm_add_ps(
            _mm_mul_ps(_mm_shuffle_ps(_val2, _val2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(oVal2, oVal2, 245))),
            _mm_mul_ps(_val2, _mm_shuffle_ps(oVal2, oVal2, 160)));
        return _val2;
    }
    inline Complex8x2Simd operator*(const float& rhs) const { return _mm_mul_ps(_val2, _mm_set1_ps(rhs)); }
    inline Complex8x2Simd operator-() const { return _mm_mul_ps(_mm_set1_ps(-1.0f), _val2); }
    inline Complex8x2Simd operator*=(const float& other)
    {
        _val2 = _mm_mul_ps(_val2, _mm_set1_ps(other));
        return _val2;
    }
};

union complex2 {
    Complex8x2Simd c2;
    std::complex<float> c[2];
    float f[4];
    complex2() {}
    complex2(const Complex8x2Simd& cm2) { c2 = cm2; }
    complex2(const std::complex<float>& cm1, const std::complex<float>& cm2)
    {
        c[0] = cm1;
        c[1] = cm2;
    }
    inline complex2 operator*(const complex2& rhs) const { return c2 * rhs.c2; }
};

inline Complex8x2Simd dupeLo(const Complex8x2Simd& cmplx2) { return _mm_shuffle_ps(cmplx2._val2, cmplx2._val2, 68); }
inline Complex8x2Simd dupeHi(const Complex8x2Simd& cmplx2) { return _mm_shuffle_ps(cmplx2._val2, cmplx2._val2, 238); }
inline Complex8x2Simd matrixMul(
    const Complex8x2Simd& mtrxCol1, const Complex8x2Simd& mtrxCol2, const Complex8x2Simd& qubit)
{
    const __m128& col1 = mtrxCol1._val2;
    const __m128& col2 = mtrxCol2._val2;
    const __m128 dupeLo = _mm_shuffle_ps(qubit._val2, qubit._val2, 68);
    const __m128 dupeHi = _mm_shuffle_ps(qubit._val2, qubit._val2, 238);
    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(col1, col1, 177),
                                     _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupeLo, dupeLo, 245))),
                          _mm_mul_ps(col1, _mm_shuffle_ps(dupeLo, dupeLo, 160))),
        _mm_add_ps(
            _mm_mul_ps(_mm_shuffle_ps(col2, col2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupeHi, dupeHi, 245))),
            _mm_mul_ps(col2, _mm_shuffle_ps(dupeHi, dupeHi, 160))));
}
inline Complex8x2Simd matrixMul(
    const float& nrm, const Complex8x2Simd& mtrxCol1, const Complex8x2Simd& mtrxCol2, const Complex8x2Simd& qubit)
{
    const __m128& col1 = mtrxCol1._val2;
    const __m128& col2 = mtrxCol2._val2;
    const __m128 dupeLo = _mm_shuffle_ps(qubit._val2, qubit._val2, 68);
    const __m128 dupeHi = _mm_shuffle_ps(qubit._val2, qubit._val2, 238);
    return _mm_mul_ps(_mm_set1_ps(nrm),
        _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(col1, col2, 177),
                                  _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupeLo, dupeLo, 245))),
                       _mm_mul_ps(mtrxCol1._val2, _mm_shuffle_ps(dupeLo, dupeLo, 160))),
            _mm_add_ps(
                _mm_mul_ps(_mm_shuffle_ps(col2, col2, 177), _mm_xor_ps(SIGNMASK, _mm_shuffle_ps(dupeHi, dupeHi, 245))),
                _mm_mul_ps(col2, _mm_shuffle_ps(dupeHi, dupeHi, 160)))));
}
inline Complex8x2Simd operator*(const float& lhs, const Complex8x2Simd& rhs)
{
    return _mm_mul_ps(_mm_set1_ps(lhs), rhs._val2);
}

inline float norm(const Complex8x2Simd& c)
{
    const complex2 cu(_mm_mul_ps(c._val2, c._val2));
    return (cu.f[0] + cu.f[1] + cu.f[2] + cu.f[3]);
}
} // namespace Qrack
