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

/// Exponentiate of arbitrary single bit gate
void QInterface::Exp(bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit, complex* matrix2x2)
{
    // Solve for the eigenvalues and eigenvectors of a 2x2 matrix, diagonalize, exponentiate, return to the original
    // basis, and apply.

    // Diagonal matrices are a special case.
    bool isDiag = true;
    if (norm(matrix2x2[0] - complex(ONE_R1, ZERO_R1)) > min_norm) {
        isDiag = false;
    } else if (norm(matrix2x2[1]) > min_norm) {
        isDiag = false;
    } else if (norm(matrix2x2[2] - complex(ONE_R1, ZERO_R1)) > min_norm) {
        isDiag = false;
    } else if (norm(matrix2x2[3]) > min_norm) {
        isDiag = false;
    }

    complex jacobian[4];
    complex invJacobian[4];

    if (!isDiag) {
        complex eVal1, eVal2;

        real1 nrm;

        complex tr = matrix2x2[0] * matrix2x2[3];
        complex det = tr - matrix2x2[1] * matrix2x2[2];
        complex sr = sqrt(tr * tr - 4.0f * det);
        eVal1 = (tr + sr) / 2.0f;
        eVal2 = (tr - sr) / 2.0f;

        jacobian[0] = matrix2x2[0] - eVal2;
        jacobian[2] = matrix2x2[3] - eVal2;

        jacobian[1] = matrix2x2[0] - eVal1;
        jacobian[3] = matrix2x2[3] - eVal1;

        nrm = norm(jacobian[0]) + norm(jacobian[2]);
        nrm = sqrt(nrm);
        jacobian[0] = jacobian[0] / nrm;
        jacobian[2] = jacobian[2] / nrm;

        nrm = norm(jacobian[1]) + norm(jacobian[3]);
        nrm = sqrt(nrm);
        jacobian[1] = jacobian[1] / nrm;
        jacobian[3] = jacobian[3] / nrm;

        det = (jacobian[0] * jacobian[2]) - (jacobian[1] * jacobian[3]);
        invJacobian[0] = jacobian[0] / det;
        invJacobian[1] = jacobian[2] / det;
        invJacobian[2] = jacobian[1] / det;
        invJacobian[3] = jacobian[3] / det;

        // TODO: Apply Jacobian and inverse to diagonalize;
    }

    complex expOfGate[4] = {
        // 2x2
        // Note: For a hermitian input gate, this should be a theoretically unitary output transformation.
        ((real1)exp(-imag(matrix2x2[0]))) * complex((real1)cos(real(matrix2x2[0])), (real1)sin(real(matrix2x2[0]))),
        complex(ZERO_R1, ZERO_R1),

        complex(ZERO_R1, ZERO_R1),
        ((real1)exp(-imag(matrix2x2[3]))) * complex((real1)cos(real(matrix2x2[3])), (real1)sin(real(matrix2x2[3]))),
    };

    if (!isDiag) {
        // TODO: Apply Jacobian and inverse to return to original basis.
    }

    ApplyControlledSingleBit(controls, controlLen, qubit, expOfGate);
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
