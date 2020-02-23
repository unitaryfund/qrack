//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <functional>
#include <memory>
#include <random>

#define bitLenInt uint8_t
#if ENABLE_PURE32
#define bitCapInt uint32_t
#define ONE_BCI 1U
#elif ENABLE_UINT128
#define bitCapInt __uint128_t
#define ONE_BCI ((__uint128_t)1U)
#else
#define bitCapInt uint64_t
#define ONE_BCI 1ULL
#endif
#define bitsInByte 8
#define qrack_rand_gen std::mt19937_64
#define qrack_rand_gen_ptr std::shared_ptr<qrack_rand_gen>
#define QRACK_ALIGN_SIZE 64

#include "config.h"

#include <complex>

#if ENABLE_COMPLEX8
namespace Qrack {
typedef std::complex<float> complex;
typedef float real1;
} // namespace Qrack
#define ZERO_R1 0.0f
#define ONE_R1 1.0f
#define PI_R1 (real1) M_PI
// min_norm is the minimum probability neighborhood to check for exactly 1 or 0 probability. Values were chosen based on
// the results of the tests in accuracy.cpp.
#define min_norm 1e-14f
#define REAL1_DEFAULT_ARG -999.0f
#else
//#include "complex16simd.hpp"
namespace Qrack {
typedef std::complex<double> complex;
typedef double real1;
} // namespace Qrack
#define ZERO_R1 0.0
#define ONE_R1 1.0
#define PI_R1 M_PI
// min_norm is the minimum probability neighborhood to check for exactly 1 or 0 probability. Values were chosen based on
// the results of the tests in accuracy.cpp.
#define min_norm 1e-30
#define REAL1_DEFAULT_ARG -999.0
#endif

#define ONE_CMPLX complex(ONE_R1, ZERO_R1)
#define ZERO_CMPLX complex(ZERO_R1, ZERO_R1)
#define I_CMPLX complex(ZERO_R1, ONE_R1)
#define CMPLX_DEFAULT_ARG complex(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG)

// approxcompare_error is the maximum acceptable sum of probability amplitude difference for ApproxCompare to return
// "true." When TrySeparate or TryDecohere is applied after the QFT followed by its inverse on a permutation, the sum of
// square errors of probability is generally less than 10^-11, for float accuracy. (A small number of trials return many
// orders larger error, but these cases should not be separated, as the code stands.)
#define approxcompare_error 1e-7f

namespace Qrack {
typedef std::shared_ptr<complex> BitOp;

/** Called once per value between begin and end. */
typedef std::function<void(const bitCapInt, const int cpu)> ParallelFunc;
typedef std::function<bitCapInt(const bitCapInt, const int cpu)> IncrementFunc;

void mul2x2(complex* left, complex* right, complex* out);
void exp2x2(complex* matrix2x2, complex* outMatrix2x2);
void log2x2(complex* matrix2x2, complex* outMatrix2x2);
} // namespace Qrack
