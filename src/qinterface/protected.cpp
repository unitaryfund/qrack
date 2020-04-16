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

#include "qinterface.hpp"
#include <algorithm>
#include <cmath>

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
#else
    return (unsigned char*)aligned_alloc(QRACK_ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE
                                                                  : (sizeof(unsigned char) * ucharCount));
#endif
}

void cl_free(void* toFree)
{
#if defined(_WIN32)
    _aligned_free(toFree);
#else
    free(toFree);
#endif
}

bitCapInt intPow(bitCapInt base, bitCapInt power)
{
    if (power == 0U) {
        return ONE_BCI;
    }
    if (power == ONE_BCI) {
        return base;
    }
    return base * intPow(base, power - 1);
}

bitCapIntOcl intPowOcl(bitCapIntOcl base, bitCapIntOcl power)
{
    if (power == 0U) {
        return ONE_BCI;
    }
    if (power == ONE_BCI) {
        return base;
    }
    return base * intPowOcl(base, power - 1);
}

void mul2x2(complex* left, complex* right, complex* out)
{
    out[0] = (left[0] * right[0]) + (left[1] * right[2]);
    out[1] = (left[0] * right[1]) + (left[1] * right[3]);
    out[2] = (left[2] * right[0]) + (left[3] * right[2]);
    out[3] = (left[2] * right[1]) + (left[3] * right[3]);
}

void _expLog2x2(complex* matrix2x2, complex* outMatrix2x2, bool isExp)
{
    // Solve for the eigenvalues and eigenvectors of a 2x2 matrix, diagonalize, exponentiate, return to the original
    // basis, and apply.

    // Diagonal matrices are a special case.
    bool isDiag = true;
    if (norm(matrix2x2[1]) != 0) {
        isDiag = false;
    } else if (norm(matrix2x2[2]) != 0) {
        isDiag = false;
    }

    complex expOfGate[4];
    complex jacobian[4];
    complex inverseJacobian[4];
    complex tempMatrix2x2[4];

    // Diagonalize the matrix, if it is not already diagonal. Otherwise, copy it into the temporary matrix.
    if (!isDiag) {
        complex trace = matrix2x2[0] + matrix2x2[3];
        complex determinant = (matrix2x2[0] * matrix2x2[3]) - (matrix2x2[1] * matrix2x2[2]);
        complex quadraticRoot = sqrt(trace * trace - (real1)(4.0) * determinant);
        complex eigenvalue1 = (trace + quadraticRoot) / (real1)2.0;
        complex eigenvalue2 = (trace - quadraticRoot) / (real1)2.0;

        jacobian[0] = matrix2x2[0] - eigenvalue1;
        jacobian[2] = matrix2x2[2];

        jacobian[1] = matrix2x2[1];
        jacobian[3] = matrix2x2[3] - eigenvalue2;

        expOfGate[0] = eigenvalue1;
        expOfGate[1] = complex(ZERO_R1, ZERO_R1);
        expOfGate[2] = complex(ZERO_R1, ZERO_R1);
        expOfGate[3] = eigenvalue2;

        real1 nrm = std::sqrt(norm(jacobian[0]) + norm(jacobian[2]));
        jacobian[0] /= nrm;
        jacobian[2] /= nrm;

        nrm = std::sqrt(norm(jacobian[1]) + norm(jacobian[3]));
        jacobian[1] /= nrm;
        jacobian[3] /= nrm;

        determinant = (jacobian[0] * jacobian[3]) - (jacobian[1] * jacobian[2]);
        inverseJacobian[0] = jacobian[3] / determinant;
        inverseJacobian[1] = -jacobian[1] / determinant;
        inverseJacobian[2] = -jacobian[2] / determinant;
        inverseJacobian[3] = jacobian[0] / determinant;
    } else {
        std::copy(matrix2x2, matrix2x2 + 4, expOfGate);
    }

    if (isExp) {
        // In this branch, we calculate e^(matrix2x2).

        // Note: For a (2x2) hermitian input gate, this theoretically produces a unitary output transformation.
        expOfGate[0] = ((real1)std::exp(real(expOfGate[0]))) *
            complex((real1)cos(imag(expOfGate[0])), (real1)sin(imag(expOfGate[0])));
        expOfGate[1] = complex(ZERO_R1, ZERO_R1);
        expOfGate[2] = complex(ZERO_R1, ZERO_R1);
        expOfGate[3] = ((real1)std::exp(real(expOfGate[3]))) *
            complex((real1)cos(imag(expOfGate[3])), (real1)sin(imag(expOfGate[3])));
    } else {
        // In this branch, we calculate log(matrix2x2).
        expOfGate[0] = complex(std::log(abs(expOfGate[0])), arg(expOfGate[0]));
        expOfGate[1] = complex(ZERO_R1, ZERO_R1);
        expOfGate[2] = complex(ZERO_R1, ZERO_R1);
        expOfGate[3] = complex(std::log(abs(expOfGate[3])), arg(expOfGate[3]));
    }

    if (!isDiag) {
        mul2x2(expOfGate, inverseJacobian, tempMatrix2x2);
        mul2x2(jacobian, tempMatrix2x2, expOfGate);
    }

    std::copy(expOfGate, expOfGate + 4, outMatrix2x2);
}

void exp2x2(complex* matrix2x2, complex* outMatrix2x2) { _expLog2x2(matrix2x2, outMatrix2x2, true); }

void log2x2(complex* matrix2x2, complex* outMatrix2x2) { _expLog2x2(matrix2x2, outMatrix2x2, false); }

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
    if (skipPowersCount == 0) {
        return perm;
    }

    bitCapInt i, iHigh, iLow;
    bitCapIntOcl p;
    iHigh = perm;
    i = 0;
    for (p = 0; p < skipPowersCount; p++) {
        iLow = iHigh & (skipPowers[p] - ONE_BCI);
        i |= iLow;
        iHigh = (iHigh ^ iLow) << ONE_BCI;
    }
    i |= iHigh;

    return i;
}

bool QInterface::IsIdentity(const complex* mtrx, bool isControlled)
{
    // If the effect of applying the buffer would be (approximately or exactly) that of applying the identity
    // operator, then we can discard this buffer without applying it.
    if ((mtrx[0] != mtrx[3]) || (norm(mtrx[1]) != 0) || (norm(mtrx[2]) != 0)) {
        return false;
    }

    // Now, we now that mtrx[1] and mtrx[2] are 0 and mtrx[0]==mtrx[3].

    // If the global phase offset has been randomized, we assume that global phase offsets are inconsequential, for
    // the user's purposes. If the global phase offset has not been randomized, user code might explicitly depend on
    // the global phase offset (but shouldn't).

    if ((isControlled || !randGlobalPhase) && (imag(mtrx[0]) != 0)) {
        return false;
    }

    // If we haven't returned false by now, we're buffering an identity operator (exactly or up to an arbitrary global
    // phase factor).
    return true;
}

#if ENABLE_UINT128
std::ostream& operator<<(std::ostream& left, __uint128_t right)
{
    // 39 decimal digits in 2^128
    unsigned char digits[39];
    int i;
    for (i = 0; i < 39; i++) {
        digits[i] = right % 10U;
        right /= 10U;
    }

    bool hasFirstDigit = false;
    for (i = 38; i >= 0; i--) {
        if (hasFirstDigit || (digits[i] > 0)) {
            left << (int)digits[i];
            hasFirstDigit = true;
        }
    }

    return left;
}
#endif

} // namespace Qrack
