//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// SIMD implementation of the double precision complex type
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <xmmintrin.h>

namespace Qrack {

	/// SIMD implementation of the double precision complex type
	/** SIMD implementation of the double precision complex type. */
	struct Complex8Simd {
		__m128 _val;

		Complex8Simd() {
		}
		
		Complex8Simd(__m128 v) {
			_val = v;
		}
		Complex8Simd(float real, float imag) {
			_val = _mm_set_ps(0.0f, 0.0f, imag, real);
		}

		Complex8Simd operator+(const Complex8Simd& other) const {
			return Complex8Simd(_mm_add_ps(_val, other._val));
		}
		Complex8Simd operator+=(const Complex8Simd& other) {
			_val = _mm_add_ps(_val, other._val);
			return Complex8Simd(_val);
		}
		Complex8Simd operator-(const Complex8Simd& other) const {
			return Complex8Simd(_mm_sub_ps(_val, other._val));
		}
		Complex8Simd operator-=(const Complex8Simd& other) {
			_val = _mm_sub_ps(_val, other._val);
			return Complex8Simd(_val);
		}
		Complex8Simd operator*(const Complex8Simd& other) const {
			__v4sf temp = (__v4sf)(other._val);
			return Complex8Simd(_mm_add_ps(
				_mm_mul_ps(_val, _mm_set_ps(0.0f, 0.0f, temp[1], -(temp[1]))),
				_mm_mul_ps(_mm_shuffle_ps(_val,_val,1), _mm_set1_ps(temp[0]))
			));
		}
		Complex8Simd operator*=(const Complex8Simd& other) {
			__v4sf temp = (__v4sf)(other._val);
			_val = _mm_add_ps(
				_mm_mul_ps(_val, _mm_set_ps(0.0f, 0.0f, temp[1], -(temp[1]))),
				_mm_mul_ps(_mm_shuffle_ps(_val,_val,1), _mm_set1_ps(temp[0]))
			);
			return Complex8Simd(_val);
		}
		Complex8Simd operator*(const float rhs) const {
			return _mm_mul_ps(_val, _mm_set1_ps(rhs));
		}
		Complex8Simd operator/(const Complex8Simd& other) const {
			__v4sf temp = (__v4sf)_mm_mul_ps(other._val, other._val);
			float denom = temp[0] + temp[1];
			temp = (__v4sf)(other._val);
			return Complex8Simd(
				_mm_div_ps(_mm_add_ps(
					_mm_mul_ps(_val, _mm_set_ps(0.0f, 0.0f, -(temp[1]), temp[1])),
					_mm_mul_ps(_mm_shuffle_ps(_val,_val,1), _mm_set1_ps(temp[0]))
				), _mm_set1_ps(denom))
			);
		}
		Complex8Simd operator/=(const Complex8Simd& other) {
			__v4sf temp = (__v4sf)_mm_mul_ps(other._val, other._val);
			float denom = temp[0] + temp[1];
			temp = (__v4sf)(other._val);
			_val = _mm_div_ps(_mm_add_ps(
				_mm_mul_ps(_val, _mm_set_ps(0.0f, 0.0f, -(temp[1]), temp[1])),
				_mm_mul_ps(_mm_shuffle_ps(_val,_val,1), _mm_set1_ps(temp[0]))
			), _mm_set1_ps(denom));
			return Complex8Simd(_val);
		}
		Complex8Simd operator/(const float rhs) const {
			return _mm_div_ps(_val, _mm_set1_ps(rhs));
		}
		Complex8Simd operator/=(const float rhs) {
			_val = _mm_div_ps(_val, _mm_set1_ps(rhs));
			return Complex8Simd(_val);
		}
	};

	static Complex8Simd operator*(const float lhs, const Complex8Simd& rhs) {
		return _mm_mul_ps(_mm_set1_ps(lhs), rhs._val);
	}
	static Complex8Simd operator/(const float lhs, const Complex8Simd& rhs) {
		__v4sf temp = (__v4sf)_mm_mul_ps(rhs._val, rhs._val);
		float denom = temp[0] + temp[1];
		temp = (__v4sf)_mm_div_ps(_mm_mul_ps(rhs._val, _mm_set1_ps(lhs)), _mm_set1_ps(denom));	
		return Complex8Simd(temp[0], -temp[1]);
	}

	double real(const Complex8Simd& cmplx) {
		return ((__v4sf)(cmplx._val))[0];
	}
	double imag(const Complex8Simd& cmplx) {
		return ((__v4sf)(cmplx._val))[1];
	}

	//double arg(const Complex16Simd& cmplx) {
	//	return atan2(imag(cmplx), real(cmplx));
	//}
	Complex8Simd conj(const Complex8Simd& cmplx) {
		return Complex8Simd(_mm_shuffle_ps(cmplx._val, _mm_sub_ps(_mm_set1_ps(0.0),cmplx._val), 2));
	}
	double norm(const Complex8Simd& cmplx) {
		__v4sf temp = (__v4sf)_mm_mul_ps(cmplx._val, cmplx._val);
		return (temp[0] + temp[1]); 
	}
	//double abs(const Complex16Simd& cmplx) {
	//	return sqrt(norm(cmplx));
	//}
	//Complex16Simd polar(const double rho, const double theta = 0) {
	//	return Complex16Simd(_mm_set1_ps(rho) * _mm_set_ps(cos(theta), sin(theta), 0.0f, 0.0f));
	//}
}
