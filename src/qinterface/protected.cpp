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

#include "common/qrack_types.hpp"
#include <algorithm>
#include <cmath>

namespace Qrack {

unsigned char* qrack_alloc(size_t ucharCount)
{
// ALIGN_SIZE is defined in common/qrack_types.hpp
#if defined(__APPLE__)
    void* toRet;
    posix_memalign(&toRet, ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount));
    return (unsigned char*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return (unsigned char*)_aligned_malloc(
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount),
        ALIGN_SIZE);
#else
    return (unsigned char*)aligned_alloc(ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount));
#endif
}

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride)
{
    while ((first < last) && (first < (last - stride))) {
        last -= stride;
        std::iter_swap(first, last);
        first += stride;
    }
}

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride)
{
    reverse(first, middle, stride);
    reverse(middle, last, stride);
    reverse(first, last, stride);
}

template void rotate<complex*>(complex* first, complex* middle, complex* last, bitCapInt stride);

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
    if (norm(matrix2x2[1]) > min_norm) {
        isDiag = false;
    } else if (norm(matrix2x2[2]) > min_norm) {
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

} // namespace Qrack
