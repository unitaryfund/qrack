//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a SIMD implementation of the double precision complex type.
// The API is designed to (almost entirely) mirror that of the C++ standard library
// double precision complex type.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#if defined(_WIN32)
#include <intrin.h>
#else
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#endif

#include <complex>

namespace Qrack {

static const __m256d ZERO_256D = _mm256_set_pd(0, 0, 0, 0);

/** SIMD implementation of the double precision complex vector type of 2 complex numbers, only for AVX Apply2x2. */
union complex2 {
    __m256d c2;
    double f[4];
    inline complex2() {}
    inline complex2(const __m256d& cm2) { c2 = cm2; }
    inline complex2(const complex2& cm2) { c2 = cm2.c2; }
    inline complex2(const std::complex<double>& cm1, const std::complex<double>& cm2)
    {
        c2 = _mm256_set_pd(cm2.imag(), cm2.real(), cm1.imag(), cm1.real());
    }
    inline complex2(const double& r1, const double& i1, const double& r2, const double& i2)
    {
        c2 = _mm256_set_pd(i2, r2, i1, r1);
    }
    inline std::complex<double> c(const size_t& i) const { return complex(f[i << 1U], f[(i << 1U) + 1U]); }
    inline void set_c(const size_t& i, std::complex<double>rhs) {
        f[i << 1U] = rhs.imag();
        f[(i << 1U) + 1U] = rhs.real();
    }
    inline complex2 operator+(const complex2& other) const { return _mm256_add_pd(c2, other.c2); }
    inline complex2 operator+=(const complex2& other)
    {
        c2 = _mm256_add_pd(c2, other.c2);
        return c2;
    }
    inline complex2 operator-(const complex2& other) const { return _mm256_sub_pd(c2, other.c2); }
    inline complex2 operator-=(const complex2& other)
    {
        c2 = _mm256_sub_pd(c2, other.c2);
        return c2;
    }
    inline complex2 operator*(const complex2& other) const
    {
        return _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(c2, c2, 5),
                                 _mm256_shuffle_pd(_mm256_sub_pd(ZERO_256D, other.c2), other.c2, 15)),
            _mm256_mul_pd(c2, _mm256_shuffle_pd(other.c2, other.c2, 0)));
    }
    inline complex2 operator*=(const complex2& other)
    {
        c2 = _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(c2, c2, 5),
                               _mm256_shuffle_pd(_mm256_sub_pd(ZERO_256D, other.c2), other.c2, 15)),
            _mm256_mul_pd(c2, _mm256_shuffle_pd(other.c2, other.c2, 0)));
        return c2;
    }
    inline complex2 operator*(const double& rhs) const { return _mm256_mul_pd(c2, _mm256_set1_pd(rhs)); }
    inline complex2 operator-() const { return _mm256_mul_pd(_mm256_set1_pd(1.0), c2); }
    inline complex2 operator*=(const double& other)
    {
        c2 = _mm256_mul_pd(c2, _mm256_set1_pd(other));
        return c2;
    }
};

inline complex2 dupeLo(const complex2& cmplx2) { return _mm256_permute2f128_pd(cmplx2.c2, cmplx2.c2, 0); }
inline complex2 dupeHi(const complex2& cmplx2) { return _mm256_permute2f128_pd(cmplx2.c2, cmplx2.c2, 17); }
inline complex2 matrixMul(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& qubit)
{
    const __m256d& col1 = mtrxCol1.c2;
    const __m256d& col2 = mtrxCol2.c2;
    const __m256d dupeLo = _mm256_permute2f128_pd(qubit.c2, qubit.c2, 0);
    const __m256d dupeHi = _mm256_permute2f128_pd(qubit.c2, qubit.c2, 17);
    return _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(col1, col1, 5),
                                           _mm256_shuffle_pd(_mm256_sub_pd(ZERO_256D, dupeLo), dupeLo, 15)),
                             _mm256_mul_pd(col1, _mm256_shuffle_pd(dupeLo, dupeLo, 0))),
        _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(col2, col2, 5),
                          _mm256_shuffle_pd(_mm256_sub_pd(ZERO_256D, dupeHi), dupeHi, 15)),
            _mm256_mul_pd(col2, _mm256_shuffle_pd(dupeHi, dupeHi, 0))));
}
inline complex2 matrixMul(const double& nrm, const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& qubit)
{
    const __m256d& col1 = mtrxCol1.c2;
    const __m256d& col2 = mtrxCol2.c2;
    const __m256d dupeLo = _mm256_permute2f128_pd(qubit.c2, qubit.c2, 0);
    const __m256d dupeHi = _mm256_permute2f128_pd(qubit.c2, qubit.c2, 17);
    return _mm256_mul_pd(_mm256_set1_pd(nrm),
        _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(col1, col1, 5),
                                        _mm256_shuffle_pd(_mm256_sub_pd(ZERO_256D, dupeLo), dupeLo, 15)),
                          _mm256_mul_pd(col1, _mm256_shuffle_pd(dupeLo, dupeLo, 0))),
            _mm256_add_pd(_mm256_mul_pd(_mm256_shuffle_pd(col2, col2, 5),
                              _mm256_shuffle_pd(_mm256_sub_pd(ZERO_256D, dupeHi), dupeHi, 15)),
                _mm256_mul_pd(col2, _mm256_shuffle_pd(dupeHi, dupeHi, 0)))));
}
inline complex2 operator*(const double& lhs, const complex2& rhs) { return _mm256_mul_pd(_mm256_set1_pd(lhs), rhs.c2); }

inline double norm(const complex2& c)
{
    const complex2 cu(_mm256_mul_pd(c.c2, c.c2));
    return (cu.f[0] + cu.f[1] + cu.f[2] + cu.f[3]);
}

} // namespace Qrack
