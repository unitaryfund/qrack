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

/// Measurement gate

/// PSEUDO-QUANTUM - Acts like a measurement gate, except with a specified forced result.
bool QInterface::ForceM(bitLenInt qubit, bool result, bool doForce, real1 nrmlzr)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    if (!doForce) {
        real1 prob = Rand();
        real1 oneChance = Prob(qubit);
        result = ((prob < oneChance) && (oneChance > ZERO_R1));
        nrmlzr = ONE_R1;
        if (result) {
            nrmlzr = oneChance;
        } else {
            nrmlzr = ONE_R1 - oneChance;
        }
    }
    if (nrmlzr > min_norm) {
        bitCapInt qPower = 1 << qubit;
        real1 angle = Rand() * 2 * M_PI;
        ApplyM(qPower, result, complex(cos(angle), sin(angle)) / (real1)(sqrt(nrmlzr)));
    } else {
        NormalizeState(ZERO_R1);
    }

    return result;
}

bool QInterface::M(bitLenInt qubit)
{
    // Measurement introduces an overall phase shift. Since it is applied to every state, this will not change the
    // status of our cached knowledge of phase separability. However, measurement could set some amplitudes to zero,
    // meaning the relative amplitude phases might only become separable in the process if they are not already.
    if (knowIsPhaseSeparable && (!isPhaseSeparable)) {
        knowIsPhaseSeparable = false;
    }
    return ForceM(qubit, false, false);
}

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

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
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

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
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

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
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

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
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

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    ApplyAntiControlled2x2(control, target, pauliX, false);
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

/// Apply controlled Pauli Y matrix to bit
void QInterface::CY(bitLenInt control, bitLenInt target)
{
    // if (qubit >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CY control bit cannot also be target.");
    const complex pauliY[4] = { complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, -ONE_R1), complex(ZERO_R1, ONE_R1),
        complex(ZERO_R1, ZERO_R1) };
    ApplyControlled2x2(control, target, pauliY, false);
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
    ApplyControlled2x2(control, target, pauliZ, false);
}

} // namespace Qrack
