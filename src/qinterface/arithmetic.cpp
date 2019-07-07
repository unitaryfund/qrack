//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
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

// Arithmetic:
void QInterface::QFTINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    // See Draper, https://arxiv.org/abs/quant-ph/0008033

    for (bitLenInt i = 0; i < length; i++) {
        for (bitLenInt j = 0; j <= i; j++) {
            if ((toAdd >> j) & 1U) {
                RT((-M_PI * 2) / intPow(2, i - j), inOutStart + i);
            }
        }
    }
}

/// Add integer (without sign)
void QInterface::INC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    QFT(inOutStart, length);
    QFTINC(toAdd, inOutStart, length);
    IQFT(inOutStart, length);
}

/// Subtract integer (without sign)
void QInterface::DEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt invToSub = (1U << length) - toSub;
    INC(invToSub, inOutStart, length);
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QInterface::DECS(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    bitCapInt invToSub = (1U << length) - toSub;
    INCS(invToSub, inOutStart, length, overflowIndex);
}

/// Subtract integer (without sign, with controls)
void QInterface::CDEC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    bitCapInt invToSub = (1U << length) - toSub;
    CINC(invToSub, inOutStart, length, controls, controlLen);
}

/// Subtract BCD integer (without sign)
void QInterface::DECBCD(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt invToSub = intPow(10U, length / 4U) - toSub;
    INCBCD(invToSub, inOutStart, length);
}

/// Quantum analog of classical "Full Adder" gate
void QInterface::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer

    // Assume outputBit is in 0 state.
    CCNOT(inputBit1, inputBit2, carryOut);
    CNOT(inputBit1, inputBit2);
    CCNOT(inputBit2, carryInSumOut, carryOut);
    CNOT(inputBit2, carryInSumOut);
    CNOT(inputBit1, inputBit2);
}

/// Quantum analog of classical "Full Subtractor" gate
void QInterface::FullSub(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    // Quantum computing is reversible! Simply perform the inverse operations in reverse order!
    // (CNOT and CCNOT are self-inverse.)

    // Assume outputBit is in 0 state.
    CNOT(inputBit1, inputBit2);
    CNOT(inputBit2, carryInSumOut);
    CCNOT(inputBit2, carryInSumOut, carryOut);
    CNOT(inputBit1, inputBit2);
    CCNOT(inputBit1, inputBit2, carryOut);
}

void QInterface::ADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry)
{
    if (length == 0) {
        return;
    }

    FullAdd(input1, input2, carry, output);

    if (length == 1) {
        Swap(carry, output);
        return;
    }

    // Otherwise, length > 1.
    bitLenInt end = length - 1U;
    for (bitLenInt i = 1; i < end; i++) {
        FullAdd(input1 + i, input2 + i, output + i, output + i + 1);
    }
    FullAdd(input1 + end, input2 + end, output + end, carry);
}

} // namespace Qrack
