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

#include "qinterface.hpp"

#include <future>

namespace Qrack {

/// "Phase shift gate" - Rotates as e^(-i*\theta/2) around |1> state
void QInterface::RT(real1 radians, bitLenInt qubit)
{
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    const complex mtrx[4] = { complex(ONE_R1, 0), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(cosine, sine) };
    ApplySingleBit(mtrx, true, qubit);
}

/// x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis
void QInterface::RX(real1 radians, bitLenInt qubit)
{
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliRX[4] = { complex(cosine, ZERO_R1), complex(ZERO_R1, -sine), complex(ZERO_R1, -sine),
        complex(cosine, ZERO_R1) };
    ApplySingleBit(pauliRX, true, qubit);
}

/// y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis
void QInterface::RY(real1 radians, bitLenInt qubit)
{
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliRY[4] = { complex(cosine, ZERO_R1), complex(-sine, ZERO_R1), complex(sine, ZERO_R1),
        complex(cosine, ZERO_R1) };
    ApplySingleBit(pauliRY, true, qubit);
}

/// z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis
void QInterface::RZ(real1 radians, bitLenInt qubit)
{
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    const complex pauliRZ[4] = { complex(cosine, -sine), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(cosine, sine) };
    ApplySingleBit(pauliRZ, true, qubit);
}

/// Exponentiate identity operator
void QInterface::Exp(real1 radians, bitLenInt qubit)
{
    complex phaseFac = complex(cos(radians), sin(radians));
    complex expIdentity[4] = { phaseFac, complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), phaseFac };
    ApplySingleBit(expIdentity, true, qubit);
}

void matrix2x2Mul(complex* left, complex* right, complex* out)
{
    out[0] = (left[0] * right[0]) + (left[1] * right[2]);
    out[1] = (left[0] * right[1]) + (left[1] * right[3]);
    out[2] = (left[2] * right[0]) + (left[3] * right[2]);
    out[3] = (left[2] * right[1]) + (left[3] * right[3]);
}

void QInterface::ExpLog(bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit, complex* matrix2x2, bool isExp)
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
        complex quadraticRoot =
            sqrt((matrix2x2[0] - matrix2x2[3]) * (matrix2x2[0] - matrix2x2[3]) - (real1)(4.0) * determinant);
        complex eigenvalue1 = (trace + quadraticRoot) / (real1)2.0;
        complex eigenvalue2 = (trace - quadraticRoot) / (real1)2.0;

        if (norm(matrix2x2[1]) > min_norm) {
            jacobian[0] = matrix2x2[1];
            jacobian[2] = eigenvalue1 - matrix2x2[0];

            jacobian[1] = matrix2x2[1];
            jacobian[3] = eigenvalue2 - matrix2x2[0];
        } else {
            jacobian[0] = eigenvalue1 - matrix2x2[3];
            jacobian[2] = matrix2x2[2];

            jacobian[1] = eigenvalue2 - matrix2x2[3];
            jacobian[3] = matrix2x2[2];
        }

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

        matrix2x2Mul(matrix2x2, jacobian, tempMatrix2x2);
        matrix2x2Mul(inverseJacobian, tempMatrix2x2, expOfGate);
    } else {
        std::copy(matrix2x2, matrix2x2 + 4, expOfGate);
    }

    if (isExp) {
        // In this branch, we calculate e^(i * matrix2x2).

        // Note: For a (2x2) hermitian input gate, this theoretically produces a unitary output transformation.
        expOfGate[0] =
            ((real1)exp(-imag(expOfGate[0]))) * complex((real1)cos(real(expOfGate[0])), (real1)sin(real(expOfGate[0])));
        expOfGate[1] = complex(ZERO_R1, ZERO_R1);
        expOfGate[2] = complex(ZERO_R1, ZERO_R1);
        expOfGate[3] =
            ((real1)exp(-imag(expOfGate[3]))) * complex((real1)cos(real(expOfGate[3])), (real1)sin(real(expOfGate[3])));
    } else {
        // In this branch, we calculate log(matrix2x2).
        expOfGate[0] = complex(log(abs(expOfGate[0])), arg(expOfGate[0]));
        expOfGate[1] = complex(ZERO_R1, ZERO_R1);
        expOfGate[2] = complex(ZERO_R1, ZERO_R1);
        expOfGate[3] = complex(log(abs(expOfGate[3])), arg(expOfGate[3]));
    }

    if (!isDiag) {
        matrix2x2Mul(expOfGate, inverseJacobian, tempMatrix2x2);
        matrix2x2Mul(jacobian, tempMatrix2x2, expOfGate);
    }

    ApplyControlledSingleBit(controls, controlLen, qubit, expOfGate);
}

/// Imaginary exponentiate of arbitrary single bit gate
void QInterface::Exp(bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit, complex* matrix2x2)
{
    ExpLog(controls, controlLen, qubit, matrix2x2, true);
}

/// Logarithm of arbitrary single bit gate
void QInterface::Log(bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit, complex* matrix2x2)
{
    ExpLog(controls, controlLen, qubit, matrix2x2, false);
}

/// Exponentiate Pauli X operator
void QInterface::ExpX(real1 radians, bitLenInt qubit)
{
    complex phaseFac = complex(cos(radians), sin(radians));
    complex expPauliX[4] = { complex(ZERO_R1, ZERO_R1), phaseFac, phaseFac, complex(ZERO_R1, ZERO_R1) };
    ApplySingleBit(expPauliX, true, qubit);
}

/// Exponentiate Pauli Y operator
void QInterface::ExpY(real1 radians, bitLenInt qubit)
{
    complex phaseFac = complex(cos(radians), sin(radians));
    complex expPauliY[4] = { complex(ZERO_R1, ZERO_R1), phaseFac * complex(ZERO_R1, -ONE_R1),
        phaseFac * complex(ZERO_R1, ONE_R1), complex(ZERO_R1, ZERO_R1) };
    ApplySingleBit(expPauliY, true, qubit);
}

/// Exponentiate Pauli Z operator
void QInterface::ExpZ(real1 radians, bitLenInt qubit)
{
    complex phaseFac = complex(cos(radians), sin(radians));
    complex expPauliZ[4] = { phaseFac, complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), -phaseFac };
    ApplySingleBit(expPauliZ, true, qubit);
}

/// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
void QInterface::CRT(real1 radians, bitLenInt control, bitLenInt target)
{
    if (control == target) {
        throw std::invalid_argument("control bit cannot also be target.");
    }

    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    const complex mtrx[4] = { complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(cosine, sine) };
    bitLenInt controls[1] = { control };
    ApplyControlledSingleBit(controls, 1, target, mtrx);
}

/// Controlled x axis rotation - if control bit is true, rotates as e^(-i*\theta/2) around Pauli x axis
void QInterface::CRX(real1 radians, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRX control bit cannot also be target.");
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliRX[4] = { complex(cosine, ZERO_R1), complex(ZERO_R1, -sine), complex(ZERO_R1, -sine),
        complex(cosine, ZERO_R1) };
    bitLenInt controls[1] = { control };
    ApplyControlledSingleBit(controls, 1, target, pauliRX);
}

/// Controlled y axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli y axis
void QInterface::CRY(real1 radians, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRY control bit cannot also be target.");
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliRY[4] = { complex(cosine, ZERO_R1), complex(-sine, ZERO_R1), complex(sine, ZERO_R1),
        complex(cosine, ZERO_R1) };
    bitLenInt controls[1] = { control };
    ApplyControlledSingleBit(controls, 1, target, pauliRY);
}

/// Controlled z axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli z axis
void QInterface::CRZ(real1 radians, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRZ control bit cannot also be target.");
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    const complex pauliRZ[4] = { complex(cosine, -sine), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(cosine, sine) };
    bitLenInt controls[1] = { control };
    ApplyControlledSingleBit(controls, 1, target, pauliRZ);
}

} // namespace Qrack
