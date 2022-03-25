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

#include "qinterface.hpp"

#if ENABLE_COMPLEX_X2
#if FPPOW == 5
#include "common/complex8x2simd.hpp"
#elif FPPOW == 6
#include "common/complex16x2simd.hpp"
#endif
#endif

#include <algorithm>

namespace Qrack {

unsigned char* cl_alloc(size_t ucharCount)
{
#if defined(__APPLE__)
    void* toRet;
    posix_memalign(&toRet, QRACK_ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE
                                                                  : (sizeof(unsigned char) * ucharCount));
    return (unsigned char*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return (unsigned char*)_aligned_malloc(((sizeof(unsigned char) * ucharCount) < QRACK_ALIGN_SIZE)
            ? QRACK_ALIGN_SIZE
            : (sizeof(unsigned char) * ucharCount),
        QRACK_ALIGN_SIZE);
#elif defined(__ANDROID__)
    return (unsigned char*)malloc(sizeof(unsigned char) * ucharCount);
#else
    return (unsigned char*)aligned_alloc(QRACK_ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE
                                                                  : (sizeof(unsigned char) * ucharCount));
#endif
}

void cl_free(void* toFree)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
    _aligned_free(toFree);
#else
    free(toFree);
#endif
}

// See https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c
#define _INTPOW(type, fn)                                                                                              \
    type fn(type base, type power)                                                                                     \
    {                                                                                                                  \
        if (power == 0U) {                                                                                             \
            return ONE_BCI;                                                                                            \
        }                                                                                                              \
        if (power == ONE_BCI) {                                                                                        \
            return base;                                                                                               \
        }                                                                                                              \
                                                                                                                       \
        type tmp = fn(base, power >> 1U);                                                                              \
        if (power & 1U) {                                                                                              \
            return base * tmp * tmp;                                                                                   \
        }                                                                                                              \
                                                                                                                       \
        return tmp * tmp;                                                                                              \
    }

_INTPOW(bitCapInt, intPow)
_INTPOW(bitCapIntOcl, intPowOcl)

#if ENABLE_COMPLEX_X2
void mul2x2(const complex* left, const complex* right, complex* out)
{
    const complex2 left0(left[0], left[2]);
    const complex2 left1(left[1], left[3]);

    complex2 col(matrixMul(left0.c2, left1.c2, complex2(right[0], right[2]).c2));
    out[0] = col.c[0];
    out[2] = col.c[1];

    col = complex2(matrixMul(left0.c2, left1.c2, complex2(right[1], right[3]).c2));
    out[1] = col.c[0];
    out[3] = col.c[1];
}
#else
void mul2x2(const complex* left, const complex* right, complex* out)
{
    out[0] = (left[0] * right[0]) + (left[1] * right[2]);
    out[1] = (left[0] * right[1]) + (left[1] * right[3]);
    out[2] = (left[2] * right[0]) + (left[3] * right[2]);
    out[3] = (left[2] * right[1]) + (left[3] * right[3]);
}
#endif

void _expLog2x2(const complex* matrix2x2, complex* outMatrix2x2, bool isExp)
{
    // Solve for the eigenvalues and eigenvectors of a 2x2 matrix, diagonalize, exponentiate, return to the original
    // basis, and apply.

    // Diagonal matrices are a special case.
    const bool isDiag = IS_NORM_0(matrix2x2[1]) && IS_NORM_0(matrix2x2[2]);

    complex expOfGate[4];
    complex jacobian[4];
    complex inverseJacobian[4];
    complex tempMatrix2x2[4];

    // Diagonalize the matrix, if it is not already diagonal. Otherwise, copy it into the temporary matrix.
    if (!isDiag) {
        complex trace = matrix2x2[0] + matrix2x2[3];
        complex determinant = (matrix2x2[0] * matrix2x2[3]) - (matrix2x2[1] * matrix2x2[2]);
        complex quadraticRoot = trace * trace - ((real1)4.0f) * determinant;
        complex qrtf((real1)real(quadraticRoot), (real1)imag(quadraticRoot));
        qrtf = sqrt(qrtf);
        quadraticRoot = complex((real1)real(qrtf), (real1)imag(qrtf));
        complex eigenvalue1 = (trace + quadraticRoot) / (real1)2.0f;
        complex eigenvalue2 = (trace - quadraticRoot) / (real1)2.0f;

        jacobian[0] = matrix2x2[0] - eigenvalue1;
        jacobian[2] = matrix2x2[2];

        jacobian[1] = matrix2x2[1];
        jacobian[3] = matrix2x2[3] - eigenvalue2;

        expOfGate[0] = eigenvalue1;
        expOfGate[1] = ZERO_CMPLX;
        expOfGate[2] = ZERO_CMPLX;
        expOfGate[3] = eigenvalue2;

        real1 nrm = (real1)std::sqrt((real1_f)(norm(jacobian[0]) + norm(jacobian[2])));
        jacobian[0] /= nrm;
        jacobian[2] /= nrm;

        nrm = (real1)std::sqrt((real1_f)(norm(jacobian[1]) + norm(jacobian[3])));
        jacobian[1] /= nrm;
        jacobian[3] /= nrm;

        determinant = (jacobian[0] * jacobian[3]) - (jacobian[1] * jacobian[2]);
        inverseJacobian[0] = jacobian[3] / determinant;
        inverseJacobian[1] = -jacobian[1] / determinant;
        inverseJacobian[2] = -jacobian[2] / determinant;
        inverseJacobian[3] = jacobian[0] / determinant;
    } else {
        expOfGate[0] = matrix2x2[0];
        expOfGate[1] = ZERO_CMPLX;
        expOfGate[2] = ZERO_CMPLX;
        expOfGate[3] = matrix2x2[3];
    }

    if (isExp) {
        // In this branch, we calculate e^(matrix2x2).

        // Note: For a (2x2) hermitian input gate, this theoretically produces a unitary output transformation.
        expOfGate[0] = ((real1)std::exp((real1_f)real(expOfGate[0]))) *
            complex((real1)cos(imag(expOfGate[0])), (real1)sin(imag(expOfGate[0])));
        expOfGate[1] = ZERO_CMPLX;
        expOfGate[2] = ZERO_CMPLX;
        expOfGate[3] = ((real1)std::exp((real1_f)real(expOfGate[3]))) *
            complex((real1)cos(imag(expOfGate[3])), (real1)sin(imag(expOfGate[3])));
    } else {
        // In this branch, we calculate log(matrix2x2).
        expOfGate[0] = complex((real1)std::log((real1_f)abs(expOfGate[0])), (real1)arg(expOfGate[0]));
        expOfGate[1] = ZERO_CMPLX;
        expOfGate[2] = ZERO_CMPLX;
        expOfGate[3] = complex((real1)std::log((real1_f)abs(expOfGate[3])), (real1)arg(expOfGate[3]));
    }

    if (!isDiag) {
        mul2x2(expOfGate, inverseJacobian, tempMatrix2x2);
        mul2x2(jacobian, tempMatrix2x2, expOfGate);
    }

    std::copy(expOfGate, expOfGate + 4, outMatrix2x2);
}

void exp2x2(const complex* matrix2x2, complex* outMatrix2x2) { _expLog2x2(matrix2x2, outMatrix2x2, true); }

void log2x2(const complex* matrix2x2, complex* outMatrix2x2) { _expLog2x2(matrix2x2, outMatrix2x2, false); }

void inv2x2(const complex* matrix2x2, complex* outMatrix2x2)
{
    const complex det = ONE_CMPLX / (matrix2x2[0] * matrix2x2[3] - matrix2x2[1] * matrix2x2[2]);
    outMatrix2x2[0] = det * matrix2x2[3];
    outMatrix2x2[1] = det * -matrix2x2[1];
    outMatrix2x2[2] = det * -matrix2x2[2];
    outMatrix2x2[3] = det * matrix2x2[0];
}

/// Check if an addition with overflow sets the flag
bool isOverflowAdd(bitCapInt inOutInt, bitCapInt inInt, const bitCapInt& signMask, const bitCapInt& lengthPower)
{
    // Both negative:
    if (inOutInt & inInt & signMask) {
        inOutInt = ((~inOutInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
        inInt = ((~inInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
        if ((inOutInt + inInt) > signMask) {
            return true;
        }
    }
    // Both positive:
    else if ((~inOutInt) & (~inInt) & signMask) {
        if ((inOutInt + inInt) >= signMask) {
            return true;
        }
    }

    return false;
}

/// Check if a subtraction with overflow sets the flag
bool isOverflowSub(bitCapInt inOutInt, bitCapInt inInt, const bitCapInt& signMask, const bitCapInt& lengthPower)
{
    // First negative:
    if (inOutInt & (~inInt) & (signMask)) {
        inOutInt = ((~inOutInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
        if ((inOutInt + inInt) > signMask)
            return true;
    }
    // First positive:
    else if ((~inOutInt) & inInt & (signMask)) {
        inInt = ((~inInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
        if ((inOutInt + inInt) >= signMask)
            return true;
    }

    return false;
}

bitCapInt pushApartBits(const bitCapInt& perm, const bitCapInt* skipPowers, const bitLenInt skipPowersCount)
{
    if (!skipPowersCount) {
        return perm;
    }

    bitCapInt iHigh = perm;
    bitCapInt i = 0;
    for (bitCapIntOcl p = 0; p < skipPowersCount; p++) {
        bitCapInt iLow = iHigh & (skipPowers[p] - ONE_BCI);
        i |= iLow;
        iHigh = (iHigh ^ iLow) << ONE_BCI;
    }
    i |= iHigh;

    return i;
}

#if ENABLE_UINT128
std::ostream& operator<<(std::ostream& left, __uint128_t right)
{
    // 39 decimal digits in 2^128
    unsigned char digits[39];
    for (int i = 0; i < 39; i++) {
        digits[i] = right % 10U;
        right /= 10U;
    }

    bool hasFirstDigit = false;
    for (int i = 38; i >= 0; i--) {
        if (hasFirstDigit || (digits[i] > 0)) {
            left << (int)digits[i];
            hasFirstDigit = true;
        }
    }

    return left;
}
#endif

} // namespace Qrack
