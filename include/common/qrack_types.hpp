//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "config.h"

#define _USE_MATH_DEFINES

#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <math.h>
#include <memory>

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)
#define IS_SAME(c1, c2) (IS_NORM_0((c1) - (c2)))
#define IS_OPPOSITE(c1, c2) (IS_NORM_0((c1) + (c2)))

#if QBCAPPOW > 6 && defined(BOOST_AVAILABLE) || QBCAPPOW > 7
#include <boost/multiprecision/cpp_int.hpp>
#endif

#if ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if (FPPOW < 5) && !defined(__arm__)
#include "half.hpp"
#endif

#if QBCAPPOW < 8
#define bitLenInt uint8_t
#elif QBCAPPOW < 16
#define bitLenInt uint16_t
#elif QBCAPPOW < 32
#define bitLenInt uint32_t
#else
#define bitLenInt uint64_t
#endif

#if FPPOW < 5
namespace Qrack {
#ifdef __arm__
typedef std::complex<__fp16> complex;
typedef __fp16 real1;
typedef float real1_f;
typedef float real1_s;
#else
#include "half.hpp"
typedef std::complex<half_float::half> complex;
typedef half_float::half real1;
typedef float real1_f;
typedef float real1_s;
#endif
#elif FPPOW < 6
namespace Qrack {
typedef std::complex<float> complex;
typedef float real1;
typedef float real1_f;
typedef float real1_s;
#elif FPPOW < 7
namespace Qrack {
typedef std::complex<double> complex;
typedef double real1;
typedef double real1_f;
typedef double real1_s;
#else
#include <boost/multiprecision/float128.hpp>
#include <quadmath.h>
namespace Qrack {
typedef std::complex<boost::multiprecision::float128> complex;
typedef boost::multiprecision::float128 real1;
typedef boost::multiprecision::float128 real1_f;
typedef double real1_s;
#endif

#if UINTPOW < 4
constexpr uint8_t ONE_BCI = 1U;
#define bitCapIntOcl uint8_t
#elif UINTPOW < 5
constexpr uint16_t ONE_BCI = 1U;
#define bitCapIntOcl uint16_t
#elif UINTPOW < 6
#define ONE_BCI 1U
#define bitCapIntOcl uint32_t
#else
#define ONE_BCI 1UL
#define bitCapIntOcl uint64_t
#endif

#if QBCAPPOW < 6
#define bitsInCap 32
#define bitCapInt uint32_t
#elif QBCAPPOW < 7
#define bitsInCap 64
#define bitCapInt uint64_t
#elif QBCAPPOW < 8
#define bitsInCap 128
#ifdef BOOST_AVAILABLE
#define bitCapInt boost::multiprecision::uint128_t
#else
#define bitCapInt __uint128_t
#endif
#else
constexpr bitLenInt bitsInCap = ((bitLenInt)1U) << (QBCAPPOW + 3U);
#define bitCapInt                                                                                                      \
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<1ULL << QBCAPPOW, 1ULL << QBCAPPOW,           \
        boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>
#endif

typedef std::shared_ptr<complex> BitOp;

// Called once per value between begin and end.
typedef std::function<void(const bitCapIntOcl&, const unsigned& cpu)> ParallelFunc;
typedef std::function<bitCapIntOcl(const bitCapIntOcl&)> IncrementFunc;
typedef std::function<bitCapInt(const bitCapInt&)> BdtFunc;
typedef std::function<void(const bitCapInt&, const unsigned& cpu)> ParallelFuncBdt;

class StateVector;
class StateVectorArray;
class StateVectorSparse;

typedef std::shared_ptr<StateVector> StateVectorPtr;
typedef std::shared_ptr<StateVectorArray> StateVectorArrayPtr;
typedef std::shared_ptr<StateVectorSparse> StateVectorSparsePtr;

typedef std::function<void(void)> DispatchFn;

class QEngine;
typedef std::shared_ptr<QEngine> QEnginePtr;

#define bitsInByte 8U
#define qrack_rand_gen std::mt19937_64
#define qrack_rand_gen_ptr std::shared_ptr<qrack_rand_gen>
#define QRACK_ALIGN_SIZE 64U

#if FPPOW < 5
#ifdef __arm__
#define ZERO_R1 0.0f
#define ZERO_R1_F 0.0f
#define ONE_R1 1.0f
#define ONE_R1_F 1.0f
constexpr real1_f PI_R1 = (real1_f)M_PI;
constexpr real1_f SQRT2_R1 = (real1_f)M_SQRT2;
constexpr real1_f SQRT1_2_R1 = (real1_f)M_SQRT1_2;
#define REAL1_DEFAULT_ARG -999.0f
// Half of the amplitude of 16 maximally superposed qubits in any permutation
#define REAL1_EPSILON 2e-17f
#else
constexpr real1 ZERO_R1 = (real1)0.0f;
#define ZERO_R1_F 0.0f
constexpr real1 ONE_R1 = (real1)1.0f;
#define ONE_R1_F 1.0f
constexpr real1_f PI_R1 = (real1_f)M_PI;
constexpr real1_f SQRT2_R1 = (real1_f)M_SQRT2;
constexpr real1_f SQRT1_2_R1 = (real1_f)M_SQRT1_2;
constexpr real1_f REAL1_DEFAULT_ARG = (real1_f)-999.0f;
// Half of the amplitude of 16 maximally superposed qubits in any permutation
constexpr real1 REAL1_EPSILON = (real1)2e-17f;
#endif
#elif FPPOW < 6
#define ZERO_R1 0.0f
#define ZERO_R1_F 0.0f
#define ONE_R1 1.0f
#define ONE_R1_F 1.0f
constexpr real1_f PI_R1 = (real1_f)M_PI;
constexpr real1_f SQRT2_R1 = (real1_f)M_SQRT2;
constexpr real1_f SQRT1_2_R1 = (real1_f)M_SQRT1_2;
#define REAL1_DEFAULT_ARG -999.0f
// Half of the amplitude of 32 maximally superposed qubits in any permutation
#define REAL1_EPSILON 2e-33f
#elif FPPOW < 7
#define ZERO_R1 0.0
#define ZERO_R1_F 0.0
#define ONE_R1 1.0
#define ONE_R1_F 1.0
#define PI_R1 M_PI
#define SQRT2_R1 M_SQRT2
#define SQRT1_2_R1 M_SQRT1_2
#define REAL1_DEFAULT_ARG -999.0
// Half of the amplitude of 64 maximally superposed qubits in any permutation
#define REAL1_EPSILON 2e-65
#else
constexpr real1 ZERO_R1 = (real1)0.0;
#define ZERO_R1_F 0.0
constexpr real1 ONE_R1 = (real1)1.0;
#define ONE_R1_F 1.0
constexpr real1_f PI_R1 = (real1_f)M_PI;
constexpr real1_f SQRT2_R1 = (real1_f)M_SQRT2;
constexpr real1_f SQRT1_2_R1 = (real1_f)M_SQRT1_2;
#define REAL1_DEFAULT_ARG -999.0
// Half of the amplitude of 64 maximally superposed qubits in any permutation
#define REAL1_EPSILON 2e-129
#endif

#if ENABLE_CUDA
#if FPPOW < 5
#define qCudaReal1 __half
#define qCudaReal2 __half2
#define qCudaReal4 __half2*
#define qCudaCmplx __half2
#define qCudaCmplx2 __half2*
#define qCudaReal1_f float
#define make_qCudaCmplx make_half2
constexpr qCudaReal1 ZERO_R1_CUDA = (qCudaReal1)0.0f;
#define REAL1_EPSILON_CUDA 2e-17f
#define PI_R1_CUDA M_PI
#elif FPPOW < 6
#define qCudaReal1 float
#define qCudaReal2 float2
#define qCudaReal4 float4
#define qCudaCmplx float2
#define qCudaCmplx2 float4
#define qCudaReal1_f float
#define make_qCudaCmplx make_float2
#define make_qCudaCmplx2 make_float4
#define ZERO_R1_CUDA 0.0f
#define REAL1_EPSILON_CUDA REAL1_EPSILON
#define PI_R1_CUDA PI_R1
#else
#define qCudaReal1 double
#define qCudaReal2 double2
#define qCudaReal4 double4
#define qCudaCmplx double2
#define qCudaCmplx2 double4
#define qCudaReal1_f double
#define make_qCudaCmplx make_double2
#define make_qCudaCmplx2 make_double4
#define ZERO_R1_CUDA 0.0
#define REAL1_EPSILON_CUDA REAL1_EPSILON
#define PI_R1_CUDA PI_R1
#endif
#endif

constexpr complex ONE_CMPLX = complex(ONE_R1, ZERO_R1);
constexpr complex ZERO_CMPLX = complex(ZERO_R1, ZERO_R1);
constexpr complex I_CMPLX = complex(ZERO_R1, ONE_R1);
constexpr complex CMPLX_DEFAULT_ARG = complex(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG);
constexpr real1 FP_NORM_EPSILON = std::numeric_limits<real1>::epsilon();
constexpr real1_f FP_NORM_EPSILON_F = std::numeric_limits<real1_f>::epsilon();
constexpr real1_f TRYDECOMPOSE_EPSILON = (real1_f)(8 * FP_NORM_EPSILON);
} // namespace Qrack
