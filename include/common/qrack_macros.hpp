//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
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

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)
#define IS_SAME(c1, c2) (IS_NORM_0((c1) - (c2)))
#define IS_OPPOSITE(c1, c2) (IS_NORM_0((c1) + (c2)))

#if UINTPOW < 4
#define ONE_BCI ((uint8_t)1U)
#define bitCapIntOcl uint8_t
#elif UINTPOW < 5
#define ONE_BCI ((uint16_t)1U)
#define bitCapIntOcl uint16_t
#elif UINTPOW < 6
#define ONE_BCI 1U
#define bitCapIntOcl uint32_t
#else
#define ONE_BCI 1UL
#define bitCapIntOcl uint64_t
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
#define bitsInCap (8U * (((bitLenInt)1U) << QBCAPPOW))
#define bitCapInt                                                                                                      \
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<1ULL << QBCAPPOW, 1ULL << QBCAPPOW,           \
        boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>
#endif

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
#define PI_R1 ((real1_f)M_PI)
#define SQRT2_R1 ((real1_f)M_SQRT2)
#define SQRT1_2_R1 ((real1_f)M_SQRT2)
#define REAL1_DEFAULT_ARG -999.0f
// Half of the amplitude of 16 maximally superposed qubits in any permutation
#define REAL1_EPSILON 2e-17f
#else
#define ZERO_R1 ((real1)0.0f)
#define ZERO_R1_F 0.0f
#define ONE_R1 ((real1)1.0f)
#define ONE_R1_F 1.0f
#define PI_R1 ((real1)M_PI)
#define SQRT2_R1 ((real1)M_SQRT2)
#define SQRT1_2_R1 ((real1)M_SQRT1_2)
#define REAL1_DEFAULT_ARG ((real1)-999.0f)
// Half of the amplitude of 16 maximally superposed qubits in any permutation
#define REAL1_EPSILON ((real1)2e-17f)
#endif
#elif FPPOW < 6
#define ZERO_R1 0.0f
#define ZERO_R1_F 0.0f
#define ONE_R1 1.0f
#define ONE_R1_F 1.0f
#define PI_R1 ((real1_f)M_PI)
#define SQRT2_R1 ((real1_f)M_SQRT2)
#define SQRT1_2_R1 ((real1_f)M_SQRT1_2)
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
#define ZERO_R1 ((real1)0.0)
#define ZERO_R1_F 0.0
#define ONE_R1 ((real1)1.0)
#define ONE_R1_F 1.0
#define PI_R1 ((real1)M_PI)
#define SQRT2_R1 ((real1)M_SQRT2)
#define SQRT1_2_R1 ((real1)M_SQRT1_2)
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
#define ZERO_R1_CUDA ((qCudaReal1)0.0f)
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

#define ONE_CMPLX complex(ONE_R1, ZERO_R1)
#define ZERO_CMPLX complex(ZERO_R1, ZERO_R1)
#define I_CMPLX complex(ZERO_R1, ONE_R1)
#define CMPLX_DEFAULT_ARG complex(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG)
#define FP_NORM_EPSILON std::numeric_limits<real1>::epsilon()
#define FP_NORM_EPSILON_F ((real1_f)FP_NORM_EPSILON)
#define TRYDECOMPOSE_EPSILON ((real1_f)(4 * FP_NORM_EPSILON))
