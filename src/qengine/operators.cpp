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

// Logic Gates:

/// "AND" compare two bits in QEngineCPU, and store result in outputBit
void QEngineCPU::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetBit(outputBit, false);
        if (inputBit1 == inputBit2) {
            CNOT(inputBit1, outputBit);
        } else {
            CCNOT(inputBit1, inputBit2, outputBit);
        }
    } else {
        throw std::invalid_argument("Invalid AND arguments.");
    }
}

/// "AND" compare two bits in QEngineCPU, and store result in outputBit
void QEngineCPU::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetReg(outputBit, length, 0);
        if (inputBit1 == inputBit2) {
            CNOT(inputBit1, outputBit, length);
        } else {
            CCNOT(inputBit1, inputBit2, outputBit, length);
        }
    } else {
        throw std::invalid_argument("Invalid AND arguments.");
    }
}

/// "AND" compare a qubit in QEngineCPU with a classical bit, and store result in outputBit
void QEngineCPU::CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (!inputClassicalBit) {
        SetBit(outputBit, false);
    } else if (inputQBit != outputBit) {
        SetBit(outputBit, false);
        CNOT(inputQBit, outputBit);
    }
}

/// "OR" compare two bits in QEngineCPU, and store result in outputBit
void QEngineCPU::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetBit(outputBit, true);
        if (inputBit1 == inputBit2) {
            AntiCNOT(inputBit1, outputBit);
        } else {
            AntiCCNOT(inputBit1, inputBit2, outputBit);
        }
    } else {
        throw std::invalid_argument("Invalid OR arguments.");
    }
}

/// "OR" compare two bits in QEngineCPU, and store result in outputBit
void QEngineCPU::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetReg(outputBit, length, (1 << length) - 1);
        if (inputBit1 == inputBit2) {
            AntiCNOT(inputBit1, outputBit, length);
        } else {
            AntiCCNOT(inputBit1, inputBit2, outputBit, length);
        }
    } else {
        throw std::invalid_argument("Invalid OR arguments.");
    }
}

/// "OR" compare a qubit in QEngineCPU with a classical bit, and store result in outputBit
void QEngineCPU::CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputClassicalBit) {
        SetBit(outputBit, true);
    } else if (inputQBit != outputBit) {
        SetBit(outputBit, false);
        CNOT(inputQBit, outputBit);
    }
}

/// "XOR" compare two bits in QEngineCPU, and store result in outputBit
void QEngineCPU::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
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
        SetBit(outputBit, false);
        CNOT(inputBit1, outputBit);
        CNOT(inputBit2, outputBit);
    }
}

/// "XOR" compare two bits in QEngineCPU, and store result in outputBit
void QEngineCPU::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit, bitLenInt length)
{
    if (((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
        SetReg(outputBit, length, 0);
        return;
    }

    if (inputBit1 == outputBit) {
        CNOT(inputBit2, outputBit, length);
    } else if (inputBit2 == outputBit) {
        CNOT(inputBit1, outputBit, length);
    } else {
        SetReg(outputBit, length, 0);
        CNOT(inputBit1, outputBit, length);
        CNOT(inputBit2, outputBit, length);
    }
}

/// "XOR" compare a qubit in QEngineCPU with a classical bit, and store result in outputBit
void QEngineCPU::CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputQBit != outputBit) {
        SetBit(outputBit, inputClassicalBit);
        CNOT(inputQBit, outputBit);
    } else if (inputClassicalBit) {
        X(outputBit);
    }
}

} // namespace Qrack
