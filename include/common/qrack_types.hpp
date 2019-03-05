//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <memory>
#include <random>

#define bitLenInt uint8_t
#if ENABLE_PURE32
#define bitCapInt uint32_t
#else
#define bitCapInt uint64_t
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
// min_norm is the minimum probability an amplitude can have before normalization floors it to identically zero. Values
// were chosen based on the results of the tests in accuracy.cpp. Successive application of Hadamard gates returns
// permutations that should be ideally 0 to less than the 10^-16 scale, for float accuracy.
#define min_norm 1e-16f
// approxcompare_error is the maximum acceptable sum of probability amplitude difference for ApproxCompare to return
// "true." When TrySeparate or TryDecohere is applied after the QFT followed by its inverse on a permutation, the sum of
// square errors of probability is generally less than 10^-11, for float accuracy. (A small number of trials return many
// orders larger error, but these cases should not be separated, as the code stands.)
#define approxcompare_error 1e-8f
#else
//#include "complex16simd.hpp"
namespace Qrack {
typedef std::complex<double> complex;
typedef double real1;
} // namespace Qrack
#define ZERO_R1 0.0
#define ONE_R1 1.0
#define PI_R1 M_PI
// Successive application of Hadamard gates returns permutations that should be ideally 0 to less than the 10^-16 scale,
// for double accuracy.
#define min_norm 1e-27
// When TrySeparate or TryDecohere is applied after the QFT followed by its inverse on a permutation, the sum of square
// errors of probability is generally less than min_norm, for double accuracy. (A small number of trials return many
// orders larger error, but these cases should not be separated, as the code stands.)
#define approxcompare_error 1e-27
#endif

namespace Qrack {
typedef std::shared_ptr<complex> BitOp;

void mul2x2(complex* left, complex* right, complex* out);
void exp2x2(complex* matrix2x2, complex* outMatrix2x2);
void log2x2(complex* matrix2x2, complex* outMatrix2x2);
} // namespace Qrack
