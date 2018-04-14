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

#include "qregister.hpp"

namespace Qrack {

/// Set individual bit to pure |0> (false) or |1> (true) state
void QUnit::SetBit(bitLenInt qubit1, bool value)
{
    if (value != M(qubit1)) {
        X(qubit1);
    }
}

/// Doubly-controlled not
void QUnit::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    // if ((control1 >= qubitCount) || (control2 >= qubitCount))
    //	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
    if (control1 == control2) {
        throw std::invalid_argument("CCNOT control bits cannot be same bit.");
    }

    if (control1 == target || control2 == target) {
        throw std::invalid_argument("CCNOT control bits cannot also be target.");
    }

    const Complex16 pauliX[4] = {
            Complex16(0.0, 0.0), Complex16(1.0, 0.0),
            Complex16(1.0, 0.0), Complex16(0.0, 0.0) };

    bitCapInt qPowers[4];
    bitCapInt qPowersSorted[3];
    qPowers[1] = 1 << control1;
    qPowersSorted[0] = qPowers[1];
    qPowers[2] = 1 << control2;
    qPowersSorted[1] = qPowers[2];
    qPowers[3] = 1 << target;
    qPowersSorted[2] = qPowers[3];
    qPowers[0] = qPowers[1] + qPowers[2] + qPowers[3];
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(qPowers[0], qPowers[1] + qPowers[2], pauliX, 3, qPowersSorted, false, false);
}

/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
void QUnit::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    // if ((control1 >= qubitCount) || (control2 >= qubitCount))
    //	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
    if (control1 == control2) {
        throw std::invalid_argument("CCNOT control bits cannot be same bit.");
    }
    if (control1 == target || control2 == target) {
        throw std::invalid_argument("CCNOT control bits cannot also be target.");
    }

    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };

    bitCapInt qPowers[4];
    bitCapInt qPowersSorted[3];
    qPowers[1] = 1 << control1;
    qPowersSorted[0] = qPowers[1];
    qPowers[2] = 1 << control2;
    qPowersSorted[1] = qPowers[2];
    qPowers[3] = 1 << target;
    qPowersSorted[2] = qPowers[3];
    qPowers[0] = qPowers[1] + qPowers[2] + qPowers[3];
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(0, qPowers[3], pauliX, 3, qPowersSorted, false, false);
}

/// Controlled not
void QUnit::CNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }

    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };
    ApplyControlled2x2(control, target, pauliX, false);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QUnit::AntiCNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }

    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };
    ApplyAntiControlled2x2(control, target, pauliX, false);
}

/// Hadamard gate
void QUnit::H(bitLenInt qubit)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("operation on bit index greater than total
    // bits.");
    const Complex16 had[4] = { Complex16(1.0 / M_SQRT2, 0.0), Complex16(1.0 / M_SQRT2, 0.0),
        Complex16(1.0 / M_SQRT2, 0.0), Complex16(-1.0 / M_SQRT2, 0.0) };
    ApplySingleBit(qubit, had, true);
}

/// Swap values of two bits in register
void QUnit::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    // if ((qubit1 >= qubitCount) || (qubit2 >= qubitCount))
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (qubit1 != qubit2) {
        const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0),
            Complex16(0.0, 0.0) };

        bitCapInt qPowers[3];
        bitCapInt qPowersSorted[2];
        qPowers[1] = 1 << qubit1;
        qPowers[2] = 1 << qubit2;
        qPowers[0] = qPowers[1] + qPowers[2];
        if (qubit1 < qubit2) {
            qPowersSorted[0] = qPowers[1];
            qPowersSorted[1] = qPowers[2];
        } else {
            qPowersSorted[0] = qPowers[2];
            qPowersSorted[1] = qPowers[1];
        }

        Apply2x2(qPowers[2], qPowers[1], pauliX, 2, qPowersSorted, false, false);
    }
}

/// NOT gate, which is also Pauli x matrix
void QUnit::X(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };
    ApplySingleBit(qubit, pauliX, false);
}

/// Apply Pauli Y matrix to bit
void QUnit::Y(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliY[4] = { Complex16(0.0, 0.0), Complex16(0.0, -1.0), Complex16(0.0, 1.0), Complex16(0.0, 0.0) };
    ApplySingleBit(qubit, pauliY, false);
}

/// Apply Pauli Z matrix to bit
void QUnit::Z(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliZ[4] = { Complex16(1.0, 0.0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(-1.0, 0.0) };
    ApplySingleBit(qubit, pauliZ, false);
}

/// Apply controlled Pauli Y matrix to bit
void QUnit::CY(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CY control bit cannot also be target.");
    const Complex16 pauliY[4] = { Complex16(0.0, 0.0), Complex16(0.0, -1.0), Complex16(0.0, 1.0), Complex16(0.0, 0.0) };
    ApplyControlled2x2(control, target, pauliY, false);
}

/// Apply controlled Pauli Z matrix to bit
void QUnit::CZ(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CZ control bit cannot also be target.");
    const Complex16 pauliZ[4] = { Complex16(1.0, 0.0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(-1.0, 0.0) };
    ApplyControlled2x2(control, target, pauliZ, false);
}

// Single register instructions:

/// Apply Hadamard gate to each bit in "length," starting from bit index "start"
void QUnit::H(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        H(start + lcv);
    }
}

/// Apply Pauli Y matrix to each bit
void QUnit::Y(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        Y(start + lcv);
    }
}

/// Apply Pauli Z matrix to each bit
void QUnit::Z(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        Z(start + lcv);
    }
}

/// Apply controlled Pauli Y matrix to each bit
void QUnit::CY(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CY(control + lcv, target + lcv);
    }
}

/// Apply controlled Pauli Z matrix to each bit
void QUnit::CZ(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CZ(control + lcv, target + lcv);
    }
}

/// Bit-parallel "CNOT" two bit ranges in QUnit, and store result in range starting at output
void QUnit::CNOT(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt length)
{
    if (inputStart1 != inputStart2) {
        for (bitLenInt i = 0; i < length; i++) {
            CNOT(inputStart1 + i, inputStart2 + i);
        }
    }
}

} // namespace Qrack
