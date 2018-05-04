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

#include "complex16simd.hpp"

namespace Qrack {

/** SIMD implementation of the double precision complex type. */
struct Complex16x2Simd {
    __m256d _val2;

    Complex16x2Simd();
    Complex16x2Simd(const __m256d& v);
    Complex16x2Simd(const double& real1, const double& imag1, const double& real2, const double& imag2);
    Complex16x2Simd operator+(const Complex16x2Simd& other) const;
    Complex16x2Simd operator+=(const Complex16x2Simd& other);
    Complex16x2Simd operator-(const Complex16x2Simd& other) const;
    Complex16x2Simd operator-=(const Complex16x2Simd& other);
    Complex16x2Simd operator*(const Complex16x2Simd& other) const;
    Complex16x2Simd operator*=(const Complex16x2Simd& other);
    Complex16x2Simd operator*(const double rhs) const;
    Complex16x2Simd operator-() const;
    Complex16x2Simd operator*=(const double& other);
};

Complex16x2Simd operator*(const double lhs, const Complex16x2Simd& rhs);
Complex16x2Simd operator/(const double lhs, const Complex16x2Simd& rhs);

Complex16x2Simd dupeLo(const Complex16x2Simd& cmplx2);
Complex16x2Simd dupeHi(const Complex16x2Simd& cmplx2);

union ComplexUnion {
    Complex16x2Simd cmplx2;
    Complex16Simd cmplx[2];

    ComplexUnion() {};
    ComplexUnion(const Complex16Simd& cmplx0, const Complex16Simd& cmplx1);
};
} // namespace Qrack
