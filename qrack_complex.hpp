//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// SIMD implementation of the double precision complex type
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <emmintrin.h>

namespace Qrack {

	/// SIMD implementation of the double precision complex type
	/** SIMD implementation of the double precision complex type. */
	struct ComplexSimd {
		__m128d _val;

		ComplexSimd() {
		}
		
		ComplexSimd(__m128d v) {
			_val = v;
		}
		ComplexSimd(double real, double imag) {
			_val = _mm_set_pd(imag, real);
		}

		ComplexSimd operator+(const ComplexSimd& other) const {
			return ComplexSimd(_mm_add_pd(_val, other._val));
		}
		ComplexSimd operator+=(const ComplexSimd& other) {
			_val = _mm_add_pd(_val, other._val);
			return ComplexSimd(_val);
		}
		ComplexSimd operator-(const ComplexSimd& other) const {
			return ComplexSimd(_mm_sub_pd(_val, other._val));
		}
		ComplexSimd operator-=(const ComplexSimd& other) {
			_val = _mm_sub_pd(_val, other._val);
			return ComplexSimd(_val);
		}
		ComplexSimd operator*(const ComplexSimd& other) const {
			__v2df temp = (__v2df)(other._val);
			return ComplexSimd(_mm_add_pd(
				_mm_mul_pd(_val, _mm_set_pd(temp[1], -(temp[1]))),
				_mm_mul_pd(_mm_shuffle_pd(_val,_val,1), _mm_set1_pd(temp[0]))
			));
		}
		ComplexSimd operator*=(const ComplexSimd& other) {
			__v2df temp = (__v2df)(other._val);
			_val = _mm_add_pd(
				_mm_mul_pd(_val, _mm_set_pd(temp[1], -(temp[1]))),
				_mm_mul_pd(_mm_shuffle_pd(_val,_val,1), _mm_set1_pd(temp[0]))
			);
			return ComplexSimd(_val);
		}
		ComplexSimd operator*(const double rhs) const {
			return _mm_mul_pd(_val, _mm_set1_pd(rhs));
		}
		ComplexSimd operator/(const ComplexSimd& other) const {
			__v2df temp = (__v2df)_mm_mul_pd(other._val, other._val);
			double denom = temp[0] + temp[1];
			temp = (__v2df)(other._val);
			return ComplexSimd(
				_mm_div_pd(_mm_add_pd(
					_mm_mul_pd(_val, _mm_set_pd(-(temp[1]), temp[1])),
					_mm_mul_pd(_mm_shuffle_pd(_val,_val,1), _mm_set1_pd(temp[0]))
				), _mm_set1_pd(denom))
			);
		}
		ComplexSimd operator/=(const ComplexSimd& other) {
			__v2df temp = (__v2df)_mm_mul_pd(other._val, other._val);
			double denom = temp[0] + temp[1];
			temp = (__v2df)(other._val);
			_val = _mm_div_pd(_mm_add_pd(
				_mm_mul_pd(_val, _mm_set_pd(-(temp[1]), temp[1])),
				_mm_mul_pd(_mm_shuffle_pd(_val,_val,1), _mm_set1_pd(temp[0]))
			), _mm_set1_pd(denom));
			return ComplexSimd(_val);
		}
		ComplexSimd operator/(const double rhs) const {
			return _mm_div_pd(_val, _mm_set1_pd(rhs));
		}
		ComplexSimd operator/=(const double rhs) {
			_val = _mm_div_pd(_val, _mm_set1_pd(rhs));
			return ComplexSimd(_val);
		}
	};

	static ComplexSimd operator*(const double lhs, const ComplexSimd& rhs) {
		return _mm_mul_pd(_mm_set1_pd(lhs), rhs._val);
	}
	static ComplexSimd operator/(const double lhs, const ComplexSimd& rhs) {
		__v2df temp = (__v2df)_mm_mul_pd(rhs._val, rhs._val);
		double denom = temp[0] + temp[1];
		temp = (__v2df)_mm_div_pd(_mm_mul_pd(rhs._val, _mm_set1_pd(lhs)), _mm_set1_pd(denom));	
		return ComplexSimd(temp[0], -temp[1]);
	}
	double real(const ComplexSimd& cmplx) {
		return ((__v2df)(cmplx._val))[0];
	}
	double imag(const ComplexSimd& cmplx) {
		return ((__v2df)(cmplx._val))[1];
	}
	double normSqrd(const ComplexSimd& cmplx) {
		__v2df temp = (__v2df)_mm_mul_pd(cmplx._val, cmplx._val);
		return (temp[0] + temp[1]); 
	}
}
