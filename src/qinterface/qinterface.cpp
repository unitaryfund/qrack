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

#include "qinterface.hpp"

namespace Qrack {

// Apply X ("not") gate to each bit in "length," starting from bit index
// "start"
void QInterface::X(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        X(start + lcv);
    }
}

/// Set individual bit to pure |0> (false) or |1> (true) state
void QInterface::SetBit(bitLenInt qubit1, bool value)
{
    if (value != M(qubit1)) {
        X(qubit1);
    }
}

// Single register instructions:

/// Apply Hadamard gate to each bit in "length," starting from bit index "start"
void QInterface::H(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        H(start + lcv);
    }
}

/// Apply Pauli Y matrix to each bit
void QInterface::Y(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        Y(start + lcv);
    }
}

/// Apply Pauli Z matrix to each bit
void QInterface::Z(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        Z(start + lcv);
    }
}

/// Apply controlled Pauli Y matrix to each bit
void QInterface::CY(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CY(control + lcv, target + lcv);
    }
}

/// Apply controlled Pauli Z matrix to each bit
void QInterface::CZ(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CZ(control + lcv, target + lcv);
    }
}

/// Bit-parallel "CNOT" two bit ranges in QInterface, and store result in range starting at output
void QInterface::CNOT(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt length)
{
    if (inputStart1 != inputStart2) {
        for (bitLenInt i = 0; i < length; i++) {
            CNOT(inputStart1 + i, inputStart2 + i);
        }
    }
}


/// "AND" compare two bit ranges in QInterface, and store result in range starting at output
void QInterface::AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
        for (bitLenInt i = 0; i < length; i++) {
            AND(inputStart1 + i, inputStart2 + i, outputStart + i);
        }
    }
}

/// "AND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLAND(qInputStart + i, cBit, outputStart + i);
    }
}

/// "OR" compare two bit ranges in QInterface, and store result in range starting at output
void QInterface::OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
        for (bitLenInt i = 0; i < length; i++) {
            OR(inputStart1 + i, inputStart2 + i, outputStart + i);
        }
    }
}

/// "OR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLOR(qInputStart + i, cBit, outputStart + i);
    }
}

/// "XOR" compare two bit ranges in QInterface, and store result in range starting at output
void QInterface::XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
        for (bitLenInt i = 0; i < length; i++) {
            XOR(inputStart1 + i, inputStart2 + i, outputStart + i);
        }
    }
}

/// "XOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QInterface::CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLXOR(qInputStart + i, cBit, outputStart + i);
    }
}

/// Arithmetic shift left, with last 2 bits as sign and carry
void QInterface::ASL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROL(shift, start, length);
            Swap(end - 1, end - 2);
            SetReg(start, shift, 0);
        }
    }
}

/// Arithmetic shift right, with last 2 bits as sign and carry
void QInterface::ASR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROR(shift, start, length);
            Swap(end - 1, end - 2);
            SetReg(end - shift, shift, 0);
        }
    }
}

/// Logical shift left, filling the extra bits with |0>
void QInterface::LSL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            ROL(shift, start, length);
            SetReg(start, shift, 0);
        }
    }
}

/// Logical shift right, filling the extra bits with |0>
void QInterface::LSR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            ROR(shift, start, length);
            SetReg(end - shift, shift, 0);
        }
    }
}

/// Quantum Fourier Transform - Apply the quantum Fourier transform to the register
void QInterface::QFT(bitLenInt start, bitLenInt length)
{
    if (length > 0) {
        bitLenInt end = start + length;
        bitLenInt i, j;
        for (i = start; i < end; i++) {
            H(i);
            for (j = 1; j < (end - i); j++) {
                CRTDyad(1, 1 << j, i + j, i);
            }
        }
    }
}

///"Phase shift gate" - Rotates each bit as e^(-i*\theta/2) around |1> state
void QInterface::RT(double radians, bitLenInt start, bitLenInt length)
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
void QInterface::RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RTDyad(numerator, denominator, start + lcv);
    }
}

/// x axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli x axis
void QInterface::RX(double radians, bitLenInt start, bitLenInt length)
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
void QInterface::RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RXDyad(numerator, denominator, start + lcv);
    }
}

/// y axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli y axis
void QInterface::RY(double radians, bitLenInt start, bitLenInt length)
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
void QInterface::RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RYDyad(numerator, denominator, start + lcv);
    }
}

/// z axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli z axis
void QInterface::RZ(double radians, bitLenInt start, bitLenInt length)
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
void QInterface::RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RZDyad(numerator, denominator, start + lcv);
    }
}

/// Controlled "phase shift gate"
void QInterface::CRT(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRT(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction "phase shift gate"
void QInterface::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRTDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled x axis rotation
void QInterface::CRX(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRX(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction x axis rotation gate - for each bit, if control bit is true, rotates target bit as as
/// e^(i*(M_PI * numerator) / denominator) around Pauli x axis
void QInterface::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRXDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled y axis rotation
void QInterface::CRY(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRY(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction y axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli y axis
void QInterface::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRYDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled z axis rotation
void QInterface::CRZ(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRZ(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction z axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli z axis
void QInterface::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRZDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

} // namespace Qrack
