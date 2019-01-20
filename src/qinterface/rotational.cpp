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

/// Uniformly controlled y axis rotation gate - Rotates as e^(-i*\theta_k/2) around Pauli y axis for each permutation
/// "k" of the control bits.
void QInterface::UniformlyControlledRY(
    const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const real1* angles)
{
    bitCapInt permCount = 1U << controlLen;
    complex* pauliRYs = new complex[4 * permCount];

    real1 cosine, sine;
    for (bitLenInt i = 0; i < permCount; i++) {
        cosine = cos(angles[i] / 2);
        sine = sin(angles[i] / 2);

        pauliRYs[0 + 4 * i] = complex(cosine, ZERO_R1);
        pauliRYs[1 + 4 * i] = complex(-sine, ZERO_R1);
        pauliRYs[2 + 4 * i] = complex(sine, ZERO_R1);
        pauliRYs[3 + 4 * i] = complex(cosine, ZERO_R1);
    }

    UniformlyControlledSingleBit(controls, controlLen, qubitIndex, pauliRYs);

    delete[] pauliRYs;
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

/// Uniformly controlled z axis rotation gate - Rotates as e^(-i*\theta_k/2) around Pauli z axis for each permutation
/// "k" of the control bits.
void QInterface::UniformlyControlledRZ(
    const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const real1* angles)
{
    bitCapInt permCount = 1U << controlLen;
    complex* pauliRZs = new complex[4 * permCount];

    real1 cosine, sine;
    for (bitLenInt i = 0; i < permCount; i++) {
        cosine = cos(angles[i] / 2);
        sine = sin(angles[i] / 2);

        pauliRZs[0 + 4 * i] = complex(cosine, -sine);
        pauliRZs[1 + 4 * i] = complex(ZERO_R1, ZERO_R1);
        pauliRZs[2 + 4 * i] = complex(ZERO_R1, ZERO_R1);
        pauliRZs[3 + 4 * i] = complex(cosine, sine);
    }

    UniformlyControlledSingleBit(controls, controlLen, qubitIndex, pauliRZs);

    delete[] pauliRZs;
}

/// Exponentiate identity operator
void QInterface::Exp(real1 radians, bitLenInt qubit)
{
    complex phaseFac = complex(cos(radians), sin(radians));
    complex expIdentity[4] = { phaseFac, complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), phaseFac };
    ApplySingleBit(expIdentity, true, qubit);
}

/// Imaginary exponentiate of arbitrary single bit gate
void QInterface::Exp(bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit, complex* matrix2x2, bool antiCtrled)
{
    complex timesI[4];
    complex toApply[4];
    for (bitLenInt i = 0; i < 4; i++) {
        timesI[i] = complex(ZERO_R1, ONE_R1) * matrix2x2[i];
    }
    Qrack::exp2x2(timesI, toApply);
    if (antiCtrled) {
        ApplyAntiControlledSingleBit(controls, controlLen, qubit, toApply);
    } else {
        ApplyControlledSingleBit(controls, controlLen, qubit, toApply);
    }
}

/// Logarithm of arbitrary single bit gate
void QInterface::Log(bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit, complex* matrix2x2, bool antiCtrled)
{
    complex toApply[4];
    Qrack::log2x2(matrix2x2, toApply);
    if (antiCtrled) {
        ApplyAntiControlledSingleBit(controls, controlLen, qubit, toApply);
    } else {
        ApplyControlledSingleBit(controls, controlLen, qubit, toApply);
    }
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
