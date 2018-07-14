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

#include <algorithm>

#include "qinterface.hpp"

namespace Qrack {

/// Set individual bit to pure |0> (false) or |1> (true) state
void QInterface::SetBit(bitLenInt qubit1, bool value)
{
    if (value != M(qubit1)) {
        X(qubit1);
    }
}

/// Swap values of two bits in register
void QInterface::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << qubit1;
    qPowers[1] = 1 << qubit2;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], pauliX, 2, qPowersSorted, false);
}

/// Doubly-controlled not
void QInterface::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    // if ((control1 >= qubitCount) || (control2 >= qubitCount))
    //	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
    if (control1 == control2) {
        throw std::invalid_argument("CCNOT control bits cannot be same bit.");
    }

    if (control1 == target || control2 == target) {
        throw std::invalid_argument("CCNOT control bits cannot also be target.");
    }

    const complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    ApplyDoublyControlled2x2(control1, control2, target, pauliX, false);
}

/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    // if ((control1 >= qubitCount) || (control2 >= qubitCount))
    //	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
    if (control1 == control2) {
        throw std::invalid_argument("CCNOT control bits cannot be same bit.");
    }
    if (control1 == target || control2 == target) {
        throw std::invalid_argument("CCNOT control bits cannot also be target.");
    }

    const complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    ApplyDoublyAntiControlled2x2(control1, control2, target, pauliX, false);
}

/// Controlled not
void QInterface::CNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }

    const complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    ApplyControlled2x2(control, target, pauliX, false);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }

    const complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    ApplyAntiControlled2x2(control, target, pauliX, false);
}

/// Hadamard gate
void QInterface::H(bitLenInt qubit)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("operation on bit index greater than total
    // bits.");
    const complex had[4] = { complex(1.0 / M_SQRT2, 0.0), complex(1.0 / M_SQRT2, 0.0), complex(1.0 / M_SQRT2, 0.0),
        complex(-1.0 / M_SQRT2, 0.0) };
    ApplySingleBit(had, true, qubit);
}

/// NOT gate, which is also Pauli x matrix
void QInterface::X(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex pauliX[4] = { complex(0.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0) };
    ApplySingleBit(pauliX, false, qubit);
}

/// Apply Pauli Y matrix to bit
void QInterface::Y(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex pauliY[4] = { complex(0.0, 0.0), complex(0.0, -1.0), complex(0.0, 1.0), complex(0.0, 0.0) };
    ApplySingleBit(pauliY, false, qubit);
}

/// Apply Pauli Z matrix to bit
void QInterface::Z(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex pauliZ[4] = { complex(1.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(-1.0, 0.0) };
    ApplySingleBit(pauliZ, false, qubit);
}

/// Apply controlled Pauli Y matrix to bit
void QInterface::CY(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CY control bit cannot also be target.");
    const complex pauliY[4] = { complex(0.0, 0.0), complex(0.0, -1.0), complex(0.0, 1.0), complex(0.0, 0.0) };
    ApplyControlled2x2(control, target, pauliY, false);
}

/// Apply controlled Pauli Z matrix to bit
void QInterface::CZ(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CZ control bit cannot also be target.");
    const complex pauliZ[4] = { complex(1.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(-1.0, 0.0) };
    ApplyControlled2x2(control, target, pauliZ, false);
}

} // namespace Qrack
