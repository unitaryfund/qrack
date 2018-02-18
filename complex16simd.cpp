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

#define CMPLX_INC
#include "complex16simd.hpp"
#include <math.h>

namespace Qrack {

	/// SIMD implementation of the double precision complex type
	/** SIMD implementation of the double precision complex type. */

	//Public method definitions:
	Complex16Simd::Complex16Simd(){}
	Complex16Simd::Complex16Simd(__m128d v) { _val = v; }
	Complex16Simd::Complex16Simd(double real, double imag) { _val = _mm_set_pd(imag, real); }
	Complex16Simd Complex16Simd::operator+(const Complex16Simd& other) const {
		return Complex16Simd(_mm_add_pd(_val, other._val));
	}
	Complex16Simd Complex16Simd::operator+=(const Complex16Simd& other) {
		_val = _mm_add_pd(_val, other._val);
		return Complex16Simd(_val);
	}
	Complex16Simd Complex16Simd::operator-(const Complex16Simd& other) const {
		return Complex16Simd(_mm_sub_pd(_val, other._val));
	}
	Complex16Simd Complex16Simd::operator-=(const Complex16Simd& other) {
		_val = _mm_sub_pd(_val, other._val);
		return Complex16Simd(_val);
	}
	Complex16Simd Complex16Simd::operator*(const Complex16Simd& other) const {
		__v2df temp = (__v2df)(other._val);
		return Complex16Simd(_mm_add_pd(
			_mm_mul_pd(_val, _mm_set_pd(temp[1], -(temp[1]))),
			_mm_mul_pd(_mm_shuffle_pd(_val,_val,1), _mm_set1_pd(temp[0]))
		));
	}
	Complex16Simd Complex16Simd::operator*=(const Complex16Simd& other) {
		__v2df temp = (__v2df)(other._val);
		_val = _mm_add_pd(
			_mm_mul_pd(_val, _mm_set_pd(temp[1], -(temp[1]))),
			_mm_mul_pd(_mm_shuffle_pd(_val,_val,1), _mm_set1_pd(temp[0]))
		);
		return Complex16Simd(_val);
	}
	Complex16Simd Complex16Simd::operator*(const double rhs) const {
		return _mm_mul_pd(_val, _mm_set1_pd(rhs));
	}
	Complex16Simd Complex16Simd::operator/(const Complex16Simd& other) const {
		__v2df temp = (__v2df)_mm_mul_pd(other._val, other._val);
		double denom = temp[0] + temp[1];
		temp = (__v2df)(other._val);
		return Complex16Simd(
			_mm_div_pd(_mm_add_pd(
				_mm_mul_pd(_val, _mm_set_pd(-(temp[1]), temp[1])),
				_mm_mul_pd(_mm_shuffle_pd(_val,_val,1), _mm_set1_pd(temp[0]))
			), _mm_set1_pd(denom))
		);
	}
	Complex16Simd Complex16Simd::operator/=(const Complex16Simd& other) {
		__v2df temp = (__v2df)_mm_mul_pd(other._val, other._val);
		double denom = temp[0] + temp[1];
		temp = (__v2df)(other._val);
		_val = _mm_div_pd(_mm_add_pd(
			_mm_mul_pd(_val, _mm_set_pd(-(temp[1]), temp[1])),
			_mm_mul_pd(_mm_shuffle_pd(_val,_val,1), _mm_set1_pd(temp[0]))
		), _mm_set1_pd(denom));
		return Complex16Simd(_val);
	}
	Complex16Simd Complex16Simd::operator/(const double rhs) const {
		return _mm_div_pd(_val, _mm_set1_pd(rhs));
	}
	Complex16Simd Complex16Simd::operator/=(const double rhs) {
		_val = _mm_div_pd(_val, _mm_set1_pd(rhs));
		return Complex16Simd(_val);
	}

	//Imperative function definitions:
	Complex16Simd operator*(const double lhs, const Complex16Simd& rhs) {
		return _mm_mul_pd(_mm_set1_pd(lhs), rhs._val);
	}
	Complex16Simd operator/(const double lhs, const Complex16Simd& rhs) {
		__v2df temp = (__v2df)_mm_mul_pd(rhs._val, rhs._val);
		double denom = temp[0] + temp[1];
		temp = (__v2df)_mm_div_pd(_mm_mul_pd(rhs._val, _mm_set1_pd(lhs)), _mm_set1_pd(denom));	
		return Complex16Simd(temp[0], -temp[1]);
	}

	double real(const Complex16Simd& cmplx) {
		return ((__v2df)(cmplx._val))[0];
	}
	double imag(const Complex16Simd& cmplx) {
		return ((__v2df)(cmplx._val))[1];
	}

	double arg(const Complex16Simd& cmplx) {
		if ((imag(cmplx) == 0.0) && (real(cmplx) == 0.0)) return 0.0;
		return atan2(imag(cmplx), real(cmplx));
	}
	Complex16Simd conj(const Complex16Simd& cmplx) {
		return Complex16Simd(_mm_shuffle_pd(cmplx._val, _mm_sub_pd(_mm_set1_pd(0.0),cmplx._val), 2));
	}
	double norm(const Complex16Simd& cmplx) {
		__v2df temp = (__v2df)_mm_mul_pd(cmplx._val, cmplx._val);
		return (temp[0] + temp[1]); 
	}
	double abs(const Complex16Simd& cmplx) {
		return sqrt(norm(cmplx));
	}
	Complex16Simd polar(const double rho, const double theta = 0) {
		return Complex16Simd(_mm_set1_pd(rho) * _mm_set_pd(cos(theta), sin(theta)));
	}
}
