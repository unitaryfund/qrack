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

#include "qrack_types.hpp"

#include <set>
#include <vector>

namespace Qrack {

#define _bi_compare_1(left) \
    if (left > 1U) { \
        return 1; \
    } \
    if (left < 1U) { \
        return -1; \
    } \
\
    return 0;

#define _bi_div_mod(left, right, quotient, rmndr) \
    if (quotient) { \
        *quotient = left / right; \
    } \
    if (rmndr) { \
        *rmndr = left % right; \
    }

#define _bi_compare(left, right) \
    if (left > right) { \
        return 1; \
    } \
    if (left < right) { \
        return -1; \
    } \
\
    return 0;

#if QBCAPPOW < 6
inline void bi_not_ip(uint32_t* left) { *left = ~(*left); }
inline void bi_and_ip(uint32_t* left, const uint32_t& right) { *left &= right; }
inline void bi_or_ip(uint32_t* left, const uint32_t& right) { *left |= right; }
inline void bi_xor_ip(uint32_t* left, const uint32_t& right) { *left ^= right; }
inline double bi_to_double(const uint32_t& in) { return (double)in; }

inline void bi_increment(uint32_t* pBigInt, const uint32_t& value) { *pBigInt += value; }
inline void bi_decrement(uint32_t* pBigInt, const uint32_t& value) { *pBigInt -= value; }

inline void bi_lshift_ip(uint32_t* left, const uint32_t& right) { *left <<= right; }
inline void bi_rshift_ip(uint32_t* left, const uint32_t& right) { *left >>= right; }

inline int bi_and_1(const uint32_t& left) { return left & 1; }

inline int bi_compare_0(const uint32_t& left) { return (int)(bool)left; }

inline int bi_compare_1(const uint32_t& left) { _bi_compare_1(left); }

inline void bi_div_mod(const uint32_t& left, const uint32_t& right, uint32_t* quotient, uint32_t* rmndr) { _bi_div_mod(left, right, quotient, rmndr) }
inline void bi_div_mod_small(const uint32_t& left, const uint32_t& right, uint32_t* quotient, uint32_t* rmndr) { _bi_div_mod(left, right, quotient, rmndr) }

inline int bi_compare(const uint32_t& left, const uint32_t& right) { _bi_compare(left, right) }
#elif QBCAPPOW < 7
inline void bi_not_ip(uint64_t* left) { *left = ~(*left); }
inline void bi_and_ip(uint64_t* left, const uint64_t& right) { *left &= right; }
inline void bi_or_ip(uint64_t* left, const uint64_t& right) { *left |= right; }
inline void bi_xor_ip(uint64_t* left, const uint64_t& right) { *left ^= right; }
inline double bi_to_double(const uint64_t& in) { return (double)in; }

inline void bi_increment(uint64_t* pBigInt, const uint64_t& value) { *pBigInt += value; }
inline void bi_decrement(uint64_t* pBigInt, const uint64_t& value) { *pBigInt -= value; }

inline void bi_lshift_ip(uint64_t* left, const uint64_t& right) { *left <<= right; }
inline void bi_rshift_ip(uint64_t* left, const uint64_t& right) { *left >>= right; }

inline int bi_and_1(const uint64_t& left) { return left & 1; }

inline int bi_compare_0(const uint64_t& left) { return (int)(bool)left; }

inline int bi_compare_1(const uint64_t& left) { _bi_compare_1(left) }

inline void bi_div_mod(const uint64_t& left, const uint64_t& right, uint64_t* quotient, uint64_t* rmndr) { _bi_div_mod(left, right, quotient, rmndr) }
inline void bi_div_mod_small(const uint64_t& left, const uint64_t& right, uint64_t* quotient, uint64_t* rmndr) { _bi_div_mod(left, right, quotient, rmndr) }

inline int bi_compare(const uint64_t& left, const uint64_t& right) { _bi_compare(left, right) }
#endif

inline bitLenInt log2Ocl(bitCapIntOcl n)
{
// Source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if QBCAPPOW < 6
    return (bitLenInt)(bitsInByte * sizeof(unsigned int) - __builtin_clz((unsigned int)n) - 1U);
#else
    return (bitLenInt)(bitsInByte * sizeof(unsigned long long) - __builtin_clzll((unsigned long long)n) - 1U);
#endif
}

#if QBCAPPOW < 7
inline int bi_log2(const uint64_t& n) { return log2Ocl(n); }
inline int bi_log2(const uint32_t& n) { return log2Ocl(n); }
#endif
inline bitLenInt log2(bitCapInt n) { return (bitLenInt)bi_log2(n); }

inline bitCapInt pow2(const bitLenInt& p) { return ONE_BCI << p; }
inline bitCapIntOcl pow2Ocl(const bitLenInt& p) { return (bitCapIntOcl)1U << p; }
inline bitCapInt pow2Mask(const bitLenInt& p)
{
    bitCapInt toRet = ONE_BCI << p;
    bi_decrement(&toRet, 1U);
    return toRet;
}
inline bitCapIntOcl pow2MaskOcl(const bitLenInt& p) { return ((bitCapIntOcl)1U << p) - 1U; }
inline bitCapInt bitSlice(const bitLenInt& bit, const bitCapInt& source) { return (ONE_BCI << bit) & source; }
inline bitCapIntOcl bitSliceOcl(const bitLenInt& bit, const bitCapIntOcl& source)
{
    return ((bitCapIntOcl)1U << bit) & source;
}
inline bitCapInt bitRegMask(const bitLenInt& start, const bitLenInt& length)
{
    bitCapInt toRet = ONE_BCI << length;
    bi_decrement(&toRet, 1U);
    bi_lshift_ip(&toRet, start);
    return toRet;
}
inline bitCapIntOcl bitRegMaskOcl(const bitLenInt& start, const bitLenInt& length)
{
    return (((bitCapIntOcl)1U << length) - 1U) << start;
}
// Source: https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
inline bool isPowerOfTwo(const bitCapInt& x)
{
    bitCapInt y = x;
    bi_decrement(&y, 1U);
    bi_and_ip(&y, x);
    return (bi_compare_0(x) != 0) && (bi_compare_0(y) == 0);
}
inline bool isPowerOfTwoOcl(const bitCapIntOcl& x) { return x && !(x & (x - 1U)); }
inline bool isBadBitRange(const bitLenInt& start, const bitLenInt& length, const bitLenInt& qubitCount)
{
    return ((start + length) > qubitCount) || ((bitLenInt)(start + length) < start);
}
inline bool isBadPermRange(const bitCapIntOcl& start, const bitCapIntOcl& length, const bitCapIntOcl& maxQPowerOcl)
{
    return ((start + length) > maxQPowerOcl) || ((bitCapIntOcl)(start + length) < start);
}
inline void ThrowIfQbIdArrayIsBad(
    const std::vector<bitLenInt>& controls, const bitLenInt& qubitCount, std::string message)
{
    std::set<bitLenInt> dupes;
    for (size_t i = 0U; i < controls.size(); ++i) {
        if (controls[i] >= qubitCount) {
            throw std::invalid_argument(message);
        }

        if (dupes.find(controls[i]) == dupes.end()) {
            dupes.insert(controls[i]);
        } else {
            throw std::invalid_argument(message + " (Found duplicate qubit indices!)");
        }
    }
}

// These are utility functions defined in qinterface/protected.cpp:
unsigned char* cl_alloc(size_t ucharCount);
void cl_free(void* toFree);
void mul2x2(complex const* left, complex const* right, complex* out);
void exp2x2(complex const* matrix2x2, complex* outMatrix2x2);
void log2x2(complex const* matrix2x2, complex* outMatrix2x2);
void inv2x2(complex const* matrix2x2, complex* outMatrix2x2);
bool isOverflowAdd(
    bitCapIntOcl inOutInt, bitCapIntOcl inInt, const bitCapIntOcl& signMask, const bitCapIntOcl& lengthPower);
bool isOverflowSub(
    bitCapIntOcl inOutInt, bitCapIntOcl inInt, const bitCapIntOcl& signMask, const bitCapIntOcl& lengthPower);
bitCapInt pushApartBits(const bitCapInt& perm, const std::vector<bitCapInt>& skipPowers);
bitCapInt intPow(bitCapInt base, bitCapInt power);
bitCapIntOcl intPowOcl(bitCapIntOcl base, bitCapIntOcl power);

std::ostream& operator<<(std::ostream& os, bitCapInt b);
std::istream& operator>>(std::istream& is, bitCapInt& b);

#if ENABLE_ENV_VARS
const real1_f _qrack_qbdt_sep_thresh = getenv("QRACK_QBDT_SEPARABILITY_THRESHOLD")
    ? (real1_f)std::stof(std::string(getenv("QRACK_QBDT_SEPARABILITY_THRESHOLD")))
    : FP_NORM_EPSILON;
#else
const real1_f _qrack_qbdt_sep_thresh = FP_NORM_EPSILON;
#endif
} // namespace Qrack
