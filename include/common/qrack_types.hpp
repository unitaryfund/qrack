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

#define _USE_MATH_DEFINES
#include "config.h"

#include <cfloat>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <memory>
#include <random>

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
#include <boost/multiprecision/cpp_int.hpp>
#define bitCapInt boost::multiprecision::uint128_t
#else
#define bitCapInt __uint128_t
#endif
#else
#define bitsInCap (8U * (((bitLenInt)1U) << QBCAPPOW))
#include <boost/multiprecision/cpp_int.hpp>
#define bitCapInt                                                                                                      \
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<1 << QBCAPPOW, 1 << QBCAPPOW,                 \
        boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>
#endif

#define bitsInByte 8U
#define qrack_rand_gen std::mt19937_64
#define qrack_rand_gen_ptr std::shared_ptr<qrack_rand_gen>
#define QRACK_ALIGN_SIZE 64U

#if FPPOW < 5
#if !defined(__arm__)
#include "half.hpp"
#endif
namespace Qrack {
#ifdef __arm__
typedef std::complex<__fp16> complex;
typedef __fp16 real1;
typedef float real1_f;
typedef float real1_s;
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
typedef std::complex<half_float::half> complex;
typedef half_float::half real1;
typedef float real1_f;
typedef float real1_s;
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
} // namespace Qrack
#elif FPPOW < 6
namespace Qrack {
typedef std::complex<float> complex;
typedef float real1;
typedef float real1_f;
typedef float real1_s;
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
} // namespace Qrack
#elif FPPOW < 7
namespace Qrack {
typedef std::complex<double> complex;
typedef double real1;
typedef double real1_f;
typedef double real1_s;
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
} // namespace Qrack
#else
#include <boost/multiprecision/float128.hpp>
#include <quadmath.h>
namespace Qrack {
typedef std::complex<boost::multiprecision::float128> complex;
typedef boost::multiprecision::float128 real1;
typedef boost::multiprecision::float128 real1_f;
typedef double real1_s;
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
// Minimum representable difference from 1
} // namespace Qrack
#endif

#define ONE_CMPLX complex(ONE_R1, ZERO_R1)
#define ZERO_CMPLX complex(ZERO_R1, ZERO_R1)
#define I_CMPLX complex(ZERO_R1, ONE_R1)
#define CMPLX_DEFAULT_ARG complex(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG)
#define FP_NORM_EPSILON std::numeric_limits<real1>::epsilon()
#define FP_NORM_EPSILON_F ((real1_f)FP_NORM_EPSILON)
#define TRYDECOMPOSE_EPSILON ((real1_f)(8 * FP_NORM_EPSILON))

namespace Qrack {
typedef std::shared_ptr<complex> BitOp;

/** Called once per value between begin and end. */
typedef std::function<void(const bitCapIntOcl&, const unsigned& cpu)> ParallelFunc;
typedef std::function<bitCapIntOcl(const bitCapIntOcl&, const unsigned& cpu)> IncrementFunc;
typedef std::function<bitCapInt(const bitCapInt&, const unsigned& cpu)> BdtFunc;

class StateVector;
class StateVectorArray;
class StateVectorSparse;

typedef std::shared_ptr<StateVector> StateVectorPtr;
typedef std::shared_ptr<StateVectorArray> StateVectorArrayPtr;
typedef std::shared_ptr<StateVectorSparse> StateVectorSparsePtr;

class QEngine;
typedef std::shared_ptr<QEngine> QEnginePtr;

// This is a buffer struct that's capable of representing controlled single bit gates and arithmetic, when subclassed.
class StateVector {
protected:
    bitCapIntOcl capacity;

public:
    bool isReadLocked;

    StateVector(bitCapIntOcl cap)
        : capacity(cap)
        , isReadLocked(true)
    {
    }
    virtual complex read(const bitCapIntOcl& i) = 0;
    virtual void write(const bitCapIntOcl& i, const complex& c) = 0;
    /// Optimized "write" that is only guaranteed to write if either amplitude is nonzero. (Useful for the result of 2x2
    /// tensor slicing.)
    virtual void write2(const bitCapIntOcl& i1, const complex& c1, const bitCapIntOcl& i2, const complex& c2) = 0;
    virtual void clear() = 0;
    virtual void copy_in(const complex* inArray) = 0;
    virtual void copy_in(const complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length) = 0;
    virtual void copy_in(StateVectorPtr copyInSv, const bitCapIntOcl srcOffset, const bitCapIntOcl dstOffset,
        const bitCapIntOcl length) = 0;
    virtual void copy_out(complex* outArray) = 0;
    virtual void copy_out(complex* copyIn, const bitCapIntOcl offset, const bitCapIntOcl length) = 0;
    virtual void copy(StateVectorPtr toCopy) = 0;
    virtual void shuffle(StateVectorPtr svp) = 0;
    virtual void get_probs(real1* outArray) = 0;
    virtual bool is_sparse() = 0;
};

inline bitCapInt pow2(const bitLenInt& p) { return (bitCapInt)ONE_BCI << p; }
inline bitCapIntOcl pow2Ocl(const bitLenInt& p) { return (bitCapIntOcl)ONE_BCI << p; }
inline bitCapInt pow2Mask(const bitLenInt& p) { return ((bitCapInt)ONE_BCI << p) - ONE_BCI; }
inline bitCapIntOcl pow2MaskOcl(const bitLenInt& p) { return ((bitCapIntOcl)ONE_BCI << p) - ONE_BCI; }
inline bitLenInt log2(bitCapInt n)
{
#if __GNUC__ && QBCAPPOW < 7
// Source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if QBCAPPOW < 6
    return (bitLenInt)(bitsInByte * sizeof(unsigned int) - __builtin_clz((unsigned int)n) - 1U);
#else
    return (bitLenInt)(bitsInByte * sizeof(unsigned long long) - __builtin_clzll((unsigned long long)n) - 1U);
#endif
#else
    bitLenInt pow = 0U;
    bitCapInt p = n >> ONE_BCI;
    while (p) {
        p >>= ONE_BCI;
        pow++;
    }
    return pow;
#endif
}
inline bitCapInt bitSlice(const bitLenInt& bit, const bitCapInt& source)
{
    return ((bitCapInt)ONE_BCI << bit) & source;
}
inline bitCapIntOcl bitSliceOcl(const bitLenInt& bit, const bitCapIntOcl& source)
{
    return ((bitCapIntOcl)ONE_BCI << bit) & source;
}
inline bitCapInt bitRegMask(const bitLenInt& start, const bitLenInt& length)
{
    return (((bitCapInt)ONE_BCI << length) - ONE_BCI) << start;
}
inline bitCapIntOcl bitRegMaskOcl(const bitLenInt& start, const bitLenInt& length)
{
    return (((bitCapIntOcl)ONE_BCI << length) - ONE_BCI) << start;
}
// Source: https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
inline bool isPowerOfTwo(const bitCapInt& x) { return (x && !(x & (x - ONE_BCI))); }

// These are utility functions defined in qinterface/protected.cpp:
unsigned char* cl_alloc(size_t ucharCount);
void cl_free(void* toFree);
void mul2x2(const complex* left, const complex* right, complex* out);
void exp2x2(const complex* matrix2x2, complex* outMatrix2x2);
void log2x2(const complex* matrix2x2, complex* outMatrix2x2);
void inv2x2(const complex* matrix2x2, complex* outMatrix2x2);
bool isOverflowAdd(bitCapInt inOutInt, bitCapInt inInt, const bitCapInt& signMask, const bitCapInt& lengthPower);
bool isOverflowSub(bitCapInt inOutInt, bitCapInt inInt, const bitCapInt& signMask, const bitCapInt& lengthPower);
bitCapInt pushApartBits(const bitCapInt& perm, const bitCapInt* skipPowers, const bitLenInt skipPowersCount);
bitCapInt intPow(bitCapInt base, bitCapInt power);
bitCapIntOcl intPowOcl(bitCapIntOcl base, bitCapIntOcl power);
#if ENABLE_UINT128
std::ostream& operator<<(std::ostream& left, __uint128_t right);
#endif
} // namespace Qrack
