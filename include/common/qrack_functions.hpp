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

inline bitCapInt pow2(const bitLenInt& p) { return ONE_BCI << p; }
inline bitCapIntOcl pow2Ocl(const bitLenInt& p) { return (bitCapIntOcl)1U << p; }
inline bitCapInt pow2Mask(const bitLenInt& p)
{
    bitCapInt toRet = ONE_BCI << p;
    bi_decrement(&toRet, 1U);
    return toRet;
}
inline bitCapIntOcl pow2MaskOcl(const bitLenInt& p) { return ((bitCapIntOcl)1U << p) - 1U; }
inline bitLenInt log2(bitCapInt n) { return (bitLenInt)bi_log2(n); }
inline bitLenInt log2Ocl(bitCapIntOcl n)
{
// Source: https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if QBCAPPOW < 6
    return (bitLenInt)(bitsInByte * sizeof(unsigned int) - __builtin_clz((unsigned int)n) - 1U);
#else
    return (bitLenInt)(bitsInByte * sizeof(unsigned long long) - __builtin_clzll((unsigned long long)n) - 1U);
#endif
}
inline bitCapInt bitSlice(const bitLenInt& bit, const bitCapInt& source)
{
    bitCapInt toRet = ONE_BCI << bit;
    bi_and_ip(&toRet, source);
    return toRet;
}
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
