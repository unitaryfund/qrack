//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qengine_cpu.hpp"

namespace Qrack {

/// "Phase shift gate" - Rotates as e^(-i*\theta/2) around |1> state
void QEngineCPU::RT(real1 radians, bitLenInt qubit)
{
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    const complex mtrx[4] = { complex(1.0, 0), complex(0.0, 0.0), complex(0.0, 0.0), complex(cosine, sine) };
    ApplySingleBit(qubit, mtrx, true);
}

/// x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis
void QEngineCPU::RX(real1 radians, bitLenInt qubit)
{
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliRX[4] = { complex(cosine, 0.0), complex(0.0, -sine), complex(0.0, -sine), complex(cosine, 0.0) };
    ApplySingleBit(qubit, pauliRX, true);
}

/// y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis
void QEngineCPU::RY(real1 radians, bitLenInt qubit)
{
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliRY[4] = { complex(cosine, 0.0), complex(-sine, 0.0), complex(sine, 0.0), complex(cosine, 0.0) };
    ApplySingleBit(qubit, pauliRY, true);
}

/// z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis
void QEngineCPU::RZ(real1 radians, bitLenInt qubit)
{
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    const complex pauliRZ[4] = { complex(cosine, -sine), complex(0.0, 0.0), complex(0.0, 0.0), complex(cosine, sine) };
    ApplySingleBit(qubit, pauliRZ, true);
}

/// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
void QEngineCPU::CRT(real1 radians, bitLenInt control, bitLenInt target)
{
    if (control == target) {
        throw std::invalid_argument("control bit cannot also be target.");
    }

    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    const complex mtrx[4] = { complex(1.0, 0), complex(0.0, 0.0), complex(0.0, 0.0), complex(cosine, sine) };
    ApplyControlled2x2(control, target, mtrx, false);
}

/// Controlled x axis rotation - if control bit is true, rotates as e^(-i*\theta/2) around Pauli x axis
void QEngineCPU::CRX(real1 radians, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRX control bit cannot also be target.");
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliRX[4] = { complex(cosine, 0.0), complex(0.0, -sine), complex(0.0, -sine), complex(cosine, 0.0) };
    ApplyControlled2x2(control, target, pauliRX, false);
}

/// Controlled y axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli y axis
void QEngineCPU::CRY(real1 radians, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRY control bit cannot also be target.");
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    complex pauliRY[4] = { complex(cosine, 0.0), complex(-sine, 0.0), complex(sine, 0.0), complex(cosine, 0.0) };
    ApplyControlled2x2(control, target, pauliRY, false);
}

/// Controlled z axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli z axis
void QEngineCPU::CRZ(real1 radians, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRZ control bit cannot also be target.");
    real1 cosine = cos(radians / 2.0);
    real1 sine = sin(radians / 2.0);
    const complex pauliRZ[4] = { complex(cosine, -sine), complex(0.0, 0.0), complex(0.0, 0.0), complex(cosine, sine) };
    ApplyControlled2x2(control, target, pauliRZ, false);
}

} // namespace Qrack
