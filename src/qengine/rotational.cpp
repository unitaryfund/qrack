//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
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
void QEngineCPU::RT(double radians, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total // bits.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    const Complex16 mtrx[4] = { Complex16(1.0, 0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(cosine, sine) };
    ApplySingleBit(qubit, mtrx, true);
}

/**
 * Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) /
 * denominator) around |1> state.
 *
 * NOTE THAT * DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void QEngineCPU::RTDyad(int numerator, int denominator, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RT((M_PI * numerator * 2) / denominator, qubit);
}

/// x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis
void QEngineCPU::RX(double radians, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    // throw std::invalid_argument("operation on bit index greater than total bits.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    Complex16 pauliRX[4] = { Complex16(cosine, 0.0), Complex16(0.0, -sine), Complex16(0.0, -sine),
        Complex16(cosine, 0.0) };
    ApplySingleBit(qubit, pauliRX, true);
}

/**
 * Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) /
 * denominator) around Pauli x axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void QEngineCPU::RXDyad(int numerator, int denominator, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RX((-M_PI * numerator * 2) / denominator, qubit);
}

/// y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis
void QEngineCPU::RY(double radians, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    Complex16 pauliRY[4] = { Complex16(cosine, 0.0), Complex16(-sine, 0.0), Complex16(sine, 0.0),
        Complex16(cosine, 0.0) };
    ApplySingleBit(qubit, pauliRY, true);
}

/**
 * Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) /
 * denominator) around Pauli y axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void QEngineCPU::RYDyad(int numerator, int denominator, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RY((-M_PI * numerator * 2) / denominator, qubit);
}

/// z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis
void QEngineCPU::RZ(double radians, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    const Complex16 pauliRZ[4] = { Complex16(cosine, -sine), Complex16(0.0, 0.0), Complex16(0.0, 0.0),
        Complex16(cosine, sine) };
    ApplySingleBit(qubit, pauliRZ, true);
}

/**
 * Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void QEngineCPU::RZDyad(int numerator, int denominator, bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RZ((-M_PI * numerator * 2) / denominator, qubit);
}

/// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
void QEngineCPU::CRT(double radians, bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("control bit cannot also be target.");
    }

    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    const Complex16 mtrx[4] = { Complex16(1.0, 0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(cosine, sine) };
    ApplyControlled2x2(control, target, mtrx, true);
}

/// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
void QEngineCPU::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    // if (target >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRTDyad control bit cannot also be target.");
    CRT((-M_PI * numerator * 2) / denominator, control, target);
}

/// Controlled x axis rotation - if control bit is true, rotates as e^(-i*\theta/2) around Pauli x axis
void QEngineCPU::CRX(double radians, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRX control bit cannot also be target.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    Complex16 pauliRX[4] = { Complex16(cosine, 0.0), Complex16(0.0, -sine), Complex16(0.0, -sine),
        Complex16(cosine, 0.0) };
    ApplyControlled2x2(control, target, pauliRX, true);
}

/**
 * Controlled dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI *
 * numerator) / denominator) around Pauli x axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void QEngineCPU::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRXDyad control bit cannot also be target.");
    CRX((-M_PI * numerator * 2) / denominator, control, target);
}

/// Controlled y axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli y axis
void QEngineCPU::CRY(double radians, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRY control bit cannot also be target.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    Complex16 pauliRY[4] = { Complex16(cosine, 0.0), Complex16(-sine, 0.0), Complex16(sine, 0.0),
        Complex16(cosine, 0.0) };
    ApplyControlled2x2(control, target, pauliRY, true);
}

/**
 * Controlled dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void QEngineCPU::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRYDyad control bit cannot also be target.");
    CRY((-M_PI * numerator * 2) / denominator, control, target);
}

/// Controlled z axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli z axis
void QEngineCPU::CRZ(double radians, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRZ control bit cannot also be target.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    const Complex16 pauliRZ[4] = { Complex16(cosine, -sine), Complex16(0.0, 0.0), Complex16(0.0, 0.0),
        Complex16(cosine, sine) };
    ApplyControlled2x2(control, target, pauliRZ, true);
}

/**
 * Controlled dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli z
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void QEngineCPU::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRZDyad control bit cannot also be target.");
    CRZ((-M_PI * numerator * 2) / denominator, control, target);
}

// Single register instructions:

///"Phase shift gate" - Rotates each bit as e^(-i*\theta/2) around |1> state
void QEngineCPU::RT(double radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RT(radians, start + lcv);
    }
}

/**
 * Dyadic fraction "phase shift gate" - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around |1> state.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void QEngineCPU::RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RTDyad(numerator, denominator, start + lcv);
    }
}

/// x axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli x axis
void QEngineCPU::RX(double radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RX(radians, start + lcv);
    }
}

/**
 * Dyadic fraction x axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli x
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QEngineCPU::RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RXDyad(numerator, denominator, start + lcv);
    }
}

/// y axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli y axis
void QEngineCPU::RY(double radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RY(radians, start + lcv);
    }
}

/**
 * Dyadic fraction y axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QEngineCPU::RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RYDyad(numerator, denominator, start + lcv);
    }
}

/// z axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli z axis
void QEngineCPU::RZ(double radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RZ(radians, start + lcv);
    }
}

/**
 * Dyadic fraction z axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void QEngineCPU::RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RZDyad(numerator, denominator, start + lcv);
    }
}

/// Controlled "phase shift gate"
void QEngineCPU::CRT(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRT(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction "phase shift gate"
void QEngineCPU::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRTDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled x axis rotation
void QEngineCPU::CRX(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRX(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction x axis rotation gate - for each bit, if control bit is true, rotates target bit as as
/// e^(i*(M_PI * numerator) / denominator) around Pauli x axis
void QEngineCPU::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRXDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled y axis rotation
void QEngineCPU::CRY(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRY(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction y axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli y axis
void QEngineCPU::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRYDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled z axis rotation
void QEngineCPU::CRZ(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRZ(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction z axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli z axis
void QEngineCPU::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRZDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

} // namespace Qrack
