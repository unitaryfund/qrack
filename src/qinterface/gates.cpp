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

/// Hadamard gate
void QInterface::H(bitLenInt qubit)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("operation on bit index greater than total
    // bits.");
    const complex had[4] = { complex(ONE_R1 / M_SQRT2, ZERO_R1), complex(ONE_R1 / M_SQRT2, ZERO_R1),
        complex(ONE_R1 / M_SQRT2, ZERO_R1), complex(-ONE_R1 / M_SQRT2, ZERO_R1) };
    ApplySingleBit(had, true, qubit);
}

/// Apply 1/4 phase rotation
void QInterface::S(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex sOp[4] = { complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(ZERO_R1, ONE_R1) };
    ApplySingleBit(sOp, false, qubit);
}

/// Apply inverse 1/4 phase rotation
void QInterface::IS(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex isOp[4] = { complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(ZERO_R1, -ONE_R1) };
    ApplySingleBit(isOp, false, qubit);
}

/// Apply 1/8 phase rotation
void QInterface::T(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex tOp[4] = { complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(M_SQRT2, M_SQRT2) };
    ApplySingleBit(tOp, false, qubit);
}

/// Apply inverse 1/8 phase rotation
void QInterface::IT(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex itOp[4] = { complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(M_SQRT2, -M_SQRT2) };
    ApplySingleBit(itOp, false, qubit);
}

/// NOT gate, which is also Pauli x matrix
void QInterface::X(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    ApplySingleBit(pauliX, false, qubit);
}

/// Apply Pauli Y matrix to bit
void QInterface::Y(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex pauliY[4] = { complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, -ONE_R1), complex(ZERO_R1, ONE_R1),
        complex(ZERO_R1, ZERO_R1) };
    ApplySingleBit(pauliY, false, qubit);
}

/// Apply Pauli Z matrix to bit
void QInterface::Z(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex pauliZ[4] = { complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(-ONE_R1, ZERO_R1) };
    ApplySingleBit(pauliZ, false, qubit);
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

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitLenInt controls[2] = { control1, control2 };
    ApplyControlledSingleBit(controls, 2, target, pauliX);
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

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitLenInt controls[2] = { control1, control2 };
    ApplyAntiControlledSingleBit(controls, 2, target, pauliX);
}

/// Controlled not
void QInterface::CNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitLenInt controls[1] = { control };
    ApplyControlledSingleBit(controls, 1, target, pauliX);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitLenInt controls[1] = { control };
    ApplyAntiControlledSingleBit(controls, 1, target, pauliX);
}

/// Apply controlled Pauli Y matrix to bit
void QInterface::CY(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CY control bit cannot also be target.");
    const complex pauliY[4] = { complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, -ONE_R1), complex(ZERO_R1, ONE_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitLenInt controls[1] = { control };
    ApplyControlledSingleBit(controls, 1, target, pauliY);
}

/// Apply controlled Pauli Z matrix to bit
void QInterface::CZ(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CZ control bit cannot also be target.");
    const complex pauliZ[4] = { complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(-ONE_R1, ZERO_R1) };
    bitLenInt controls[1] = { control };
    ApplyControlledSingleBit(controls, 1, target, pauliZ);
}

} // namespace Qrack
