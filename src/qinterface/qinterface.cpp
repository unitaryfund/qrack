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
    /*
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        X(start);
        return;
    }

    // As a fundamental gate, the register-wise X could proceed like so:
    // for (bitLenInt lcv = 0; lcv < length; lcv++) {
    //    X(start + lcv);
    //}

    // Basically ALL register-wise gates proceed by essentially the same
    // algorithm as this simple X gate.

    // We first form bit masks for those qubits involved in the operation, and
    // those not involved in the operation. We might have more than one
    // register involved in the operation in general, but we only have one, in
    // this case.
    bitCapInt inOutMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ inOutMask;

    // Sometimes we transform the state in place. Alternatively, we often
    // allocate a new permutation state vector to transfer old probabilities
    // and phases into.
    Complex16 *nStateVec = new Complex16[maxQPower];

    // This function call is a parallel "for" loop. We have several variants of
    // the parallel for loop. Some skip certain permutations in order to
    // optimize. Some take a new permutation state vector for output, and some
    // just transform the permutation state vector in place.
    par_for(0, maxQPower, [&](const bitCapInt lcv) {
        // Set nStateVec, indexed by the loop control variable (lcv) with
        // the X'ed bits inverted, with the value of stateVec indexed by
        // lcv.

        // This is the body of the parallel "for" loop. We iterate over
        // permutations of bits.  We're going to transform from input
        // permutation state to output permutation state, and transfer the
        // probability and phase of the input permutation to the output
        // permutation.  These are the bits that aren't involved in the
        // operation.
        bitCapInt otherRes = (lcv & otherMask);

        // These are the bits in the register that is being operated on. In
        // all permutation states, the bits acted on by the gate should be
        // transformed in the logically appropriate way from input
        // permutation to output permutation. Since this is an X gate, we
        // take the involved bits and bitwise NOT them.
        bitCapInt inOutRes = ((~lcv) & inOutMask);

        // Now, we just transfer the untransformed input state's phase and
        // probability to the transformed output state.
        nStateVec[inOutRes | otherRes] = stateVec[lcv];

        // For other operations, like the quantum equivalent of a logical
        // "AND," we might have two input registers and one output
        // register. The transformation would be that we use bit masks to
        // bitwise "AND" the input values in every permutation and place
        // this logical result into the output register with another bit
        // mask, for every possible permutation state. Basically all the
        // register-wise operations in Qrack proceed this same way.
    });
    // We replace our old permutation state vector with the new one we just
    // filled, at the end.
    ResetStateVec(nStateVec);
    */

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

} // namespace Qrack
