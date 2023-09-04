//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
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

// Logic Gates:
void QInterface::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 == outputBit) || (inputBit2 == outputBit)) {
        throw std::invalid_argument("Invalid AND arguments.");
    }

    if (inputBit1 == inputBit2) {
        CNOT(inputBit1, outputBit);
    } else {
        CCNOT(inputBit1, inputBit2, outputBit);
    }
}

void QInterface::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 == outputBit) || (inputBit2 == outputBit)) {
        throw std::invalid_argument("Invalid OR arguments.");
    }

    X(outputBit);
    if (inputBit1 == inputBit2) {
        AntiCNOT(inputBit1, outputBit);
    } else {
        AntiCCNOT(inputBit1, inputBit2, outputBit);
    }
}

void QInterface::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    if (((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
        SetBit(outputBit, false);
        return;
    }

    if (inputBit1 == outputBit) {
        CNOT(inputBit2, outputBit);
    } else if (inputBit2 == outputBit) {
        CNOT(inputBit1, outputBit);
    } else {
        CNOT(inputBit1, outputBit);
        CNOT(inputBit2, outputBit);
    }
}

void QInterface::NAND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    AND(inputBit1, inputBit2, outputBit);
    X(outputBit);
}

void QInterface::NOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    OR(inputBit1, inputBit2, outputBit);
    X(outputBit);
}

void QInterface::XNOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    XOR(inputBit1, inputBit2, outputBit);
    X(outputBit);
}

void QInterface::CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputClassicalBit && (inputQBit != outputBit)) {
        CNOT(inputQBit, outputBit);
    }
}

void QInterface::CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputClassicalBit) {
        X(outputBit);
    } else if (inputQBit != outputBit) {
        CNOT(inputQBit, outputBit);
    }
}

void QInterface::CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputQBit != outputBit) {
        if (inputClassicalBit) {
            X(outputBit);
        }
        CNOT(inputQBit, outputBit);
    } else if (inputClassicalBit) {
        X(outputBit);
    }
}

void QInterface::CLNAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    CLAND(inputQBit, inputClassicalBit, outputBit);
    X(outputBit);
}

void QInterface::CLNOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    CLOR(inputQBit, inputClassicalBit, outputBit);
    X(outputBit);
}

void QInterface::CLXNOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    CLXOR(inputQBit, inputClassicalBit, outputBit);
    X(outputBit);
}

} // namespace Qrack
