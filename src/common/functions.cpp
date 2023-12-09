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

#include "qrack_functions.hpp"

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
bitCapInt intPow(bitCapInt base, bitCapInt power)
{
    if (bi_compare_0(power) == 0U) {
        return ONE_BCI;
    }
    if (bi_compare_1(power) == 0U) {
        return base;
    }

    bitCapInt tmp = intPow(base, power >> 1U);
    tmp = tmp * tmp;
    if (bi_and_1(power)) {
        tmp = tmp * base;
    }

    return tmp;
}
bitCapIntOcl intPowOcl(bitCapIntOcl base, bitCapIntOcl power)
{
    if (power == 0U) {
        return 1U;
    }
    if (power == 1U) {
        return base;
    }

    bitCapIntOcl tmp = intPowOcl(base, power >> 1U);
    tmp *= tmp;
    if (power & 1U) {
        tmp *= base;
    }

    return tmp;
}

#if ENABLE_COMPLEX_X2
void mul2x2(complex const* left, complex const* right, complex* out)
{
    const complex2 left0(left[0U], left[2U]);
    const complex2 left1(left[1U], left[3U]);
    const complex2 shuf0 = mtrxColShuff(left0);
    const complex2 shuf1 = mtrxColShuff(left1);

    complex2 col(matrixMul(left0.c2, left1.c2, shuf0.c2, shuf1.c2, complex2(right[0U], right[2U]).c2));
    out[0U] = col.c(0U);
    out[2U] = col.c(1U);

    col = complex2(matrixMul(left0.c2, left1.c2, shuf0.c2, shuf1.c2, complex2(right[1U], right[3U]).c2));
    out[1U] = col.c(0U);
    out[3U] = col.c(1U);
}
#else
void mul2x2(complex const* left, complex const* right, complex* out)
{
    out[0U] = (left[0U] * right[0U]) + (left[1U] * right[2U]);
    out[1U] = (left[0U] * right[1U]) + (left[1U] * right[3U]);
    out[2U] = (left[2U] * right[0U]) + (left[3U] * right[2U]);
    out[3U] = (left[2U] * right[1U]) + (left[3U] * right[3U]);
}
#endif

void _expLog2x2(complex const* matrix2x2, complex* outMatrix2x2, bool isExp)
{
    // Solve for the eigenvalues and eigenvectors of a 2x2 matrix, diagonalize, exponentiate, return to the original
    // basis, and apply.

    // Diagonal matrices are a special case.
    const bool isDiag = IS_NORM_0(matrix2x2[1U]) && IS_NORM_0(matrix2x2[2U]);

    complex expOfGate[4U];
    complex jacobian[4U];
    complex inverseJacobian[4U];
    complex tempMatrix2x2[4U];

    // Diagonalize the matrix, if it is not already diagonal. Otherwise, copy it into the temporary matrix.
    if (!isDiag) {
        complex trace = matrix2x2[0U] + matrix2x2[3U];
        complex determinant = (matrix2x2[0U] * matrix2x2[3U]) - (matrix2x2[1U] * matrix2x2[2U]);
        complex quadraticRoot = trace * trace - ((real1)4.0f) * determinant;
        std::complex<real1_f> qrtf((real1_f)real(quadraticRoot), (real1_f)imag(quadraticRoot));
        qrtf = sqrt(qrtf);
        quadraticRoot = complex((real1)real(qrtf), (real1)imag(qrtf));
        complex eigenvalue1 = (trace + quadraticRoot) / (real1)2.0f;
        complex eigenvalue2 = (trace - quadraticRoot) / (real1)2.0f;

        jacobian[0U] = matrix2x2[0U] - eigenvalue1;
        jacobian[2U] = matrix2x2[2U];

        jacobian[1U] = matrix2x2[1U];
        jacobian[3U] = matrix2x2[3U] - eigenvalue2;

        expOfGate[0U] = eigenvalue1;
        expOfGate[1U] = ZERO_CMPLX;
        expOfGate[2U] = ZERO_CMPLX;
        expOfGate[3U] = eigenvalue2;

        real1 nrm = (real1)std::sqrt((real1_s)(norm(jacobian[0U]) + norm(jacobian[2U])));
        jacobian[0U] /= nrm;
        jacobian[2U] /= nrm;

        nrm = (real1)std::sqrt((real1_s)(norm(jacobian[1U]) + norm(jacobian[3U])));
        jacobian[1U] /= nrm;
        jacobian[3U] /= nrm;

        determinant = (jacobian[0U] * jacobian[3U]) - (jacobian[1U] * jacobian[2U]);
        inverseJacobian[0U] = jacobian[3U] / determinant;
        inverseJacobian[1U] = -jacobian[1U] / determinant;
        inverseJacobian[2U] = -jacobian[2U] / determinant;
        inverseJacobian[3U] = jacobian[0U] / determinant;
    } else {
        expOfGate[0U] = matrix2x2[0U];
        expOfGate[1U] = ZERO_CMPLX;
        expOfGate[2U] = ZERO_CMPLX;
        expOfGate[3U] = matrix2x2[3U];
    }

    if (isExp) {
        // In this branch, we calculate e^(matrix2x2).

        // Note: For a (2x2) hermitian input gate, this theoretically produces a unitary output transformation.
        expOfGate[0U] = ((real1)std::exp((real1_s)real(expOfGate[0U]))) *
            complex((real1)cos(imag(expOfGate[0U])), (real1)sin(imag(expOfGate[0U])));
        expOfGate[1U] = ZERO_CMPLX;
        expOfGate[2U] = ZERO_CMPLX;
        expOfGate[3U] = ((real1)std::exp((real1_s)real(expOfGate[3U]))) *
            complex((real1)cos(imag(expOfGate[3U])), (real1)sin(imag(expOfGate[3U])));
    } else {
        // In this branch, we calculate log(matrix2x2).
        expOfGate[0U] = complex((real1)std::log((real1_s)abs(expOfGate[0U])), (real1)arg(expOfGate[0U]));
        expOfGate[1U] = ZERO_CMPLX;
        expOfGate[2U] = ZERO_CMPLX;
        expOfGate[3U] = complex((real1)std::log((real1_s)abs(expOfGate[3U])), (real1)arg(expOfGate[3U]));
    }

    if (!isDiag) {
        mul2x2(expOfGate, inverseJacobian, tempMatrix2x2);
        mul2x2(jacobian, tempMatrix2x2, expOfGate);
    }

    std::copy(expOfGate, expOfGate + 4U, outMatrix2x2);
}

void exp2x2(complex const* matrix2x2, complex* outMatrix2x2) { _expLog2x2(matrix2x2, outMatrix2x2, true); }

void log2x2(complex const* matrix2x2, complex* outMatrix2x2) { _expLog2x2(matrix2x2, outMatrix2x2, false); }

void inv2x2(complex const* matrix2x2, complex* outMatrix2x2)
{
    const complex det = ONE_CMPLX / (matrix2x2[0U] * matrix2x2[3U] - matrix2x2[1U] * matrix2x2[2U]);
    outMatrix2x2[0U] = det * matrix2x2[3U];
    outMatrix2x2[1U] = det * -matrix2x2[1U];
    outMatrix2x2[2U] = det * -matrix2x2[2U];
    outMatrix2x2[3U] = det * matrix2x2[0U];
}

/// Check if an addition with overflow sets the flag
bool isOverflowAdd(bitCapIntOcl inOutInt, bitCapIntOcl inInt, const bitCapIntOcl& signMask, const bitCapIntOcl& lengthPower)
{
    // Both negative:
    if (inOutInt & inInt & signMask) {
        inOutInt = ((~inOutInt) & (lengthPower - 1U)) + 1U;
        inInt = ((~inInt) & (lengthPower - 1U)) + 1U;
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
bool isOverflowSub(bitCapIntOcl inOutInt, bitCapIntOcl inInt, const bitCapIntOcl& signMask, const bitCapIntOcl& lengthPower)
{
    // First negative:
    if (inOutInt & (~inInt) & (signMask)) {
        inOutInt = ((~inOutInt) & (lengthPower - 1U)) + 1U;
        if ((inOutInt + inInt) > signMask)
            return true;
    }
    // First positive:
    else if ((~inOutInt) & inInt & (signMask)) {
        inInt = ((~inInt) & (lengthPower - 1U)) + 1U;
        if ((inOutInt + inInt) >= signMask)
            return true;
    }

    return false;
}

bitCapInt pushApartBits(const bitCapInt& perm, const std::vector<bitCapInt>& skipPowers)
{
    bitCapInt iHigh = perm;
    bitCapInt i = ZERO_BCI;
    for (size_t p = 0U; p < skipPowers.size(); ++p) {
        bitCapInt iLow = skipPowers[p];
        bi_decrement(&iLow, 1U);
        bi_and_ip(&iLow, iHigh);
        bi_or_ip(&i, iLow);
        bi_xor_ip(&iHigh, iLow);
        bi_lshift_ip(&iHigh, 1U);
    }
    bi_or_ip(&i, iHigh);

    return i;
}

} // namespace Qrack
