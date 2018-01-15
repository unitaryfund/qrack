//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a SIMD implementation of the double precision complex type.
// The API is designed to (almost entirely) mirror that of the C++ standard library
// double precision complex type.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <emmintrin.h>

namespace Qrack {

	/// SIMD implementation of the double precision complex type
	/** SIMD implementation of the double precision complex type. */
	struct Complex16Simd {
		__m128d _val;

		Complex16Simd();		
		Complex16Simd(__m128d v);
		Complex16Simd(double real, double imag);
		Complex16Simd operator+(const Complex16Simd& other) const;
		Complex16Simd operator+=(const Complex16Simd& other);
		Complex16Simd operator-(const Complex16Simd& other) const;
		Complex16Simd operator-=(const Complex16Simd& other);
		Complex16Simd operator*(const Complex16Simd& other) const;
		Complex16Simd operator*=(const Complex16Simd& other);
		Complex16Simd operator*(const double rhs) const;
		Complex16Simd operator/(const Complex16Simd& other) const;
		Complex16Simd operator/=(const Complex16Simd& other);
		Complex16Simd operator/(const double rhs) const;
		Complex16Simd operator/=(const double rhs);
	};

	Complex16Simd operator*(const double lhs, const Complex16Simd& rhs);
	Complex16Simd operator/(const double lhs, const Complex16Simd& rhs);

	double real(const Complex16Simd& cmplx);
	double imag(const Complex16Simd& cmplx);
	double arg(const Complex16Simd& cmplx);
	Complex16Simd conj(const Complex16Simd& cmplx);
	double norm(const Complex16Simd& cmplx);
	double abs(const Complex16Simd& cmplx);
	Complex16Simd polar(const double rho, const double theta);
}
