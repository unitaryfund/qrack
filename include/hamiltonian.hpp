//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <memory>
#include <vector>

namespace Qrack {

/**
 * A Hamiltonian can be specified in terms of arbitrary controlled single bit gates, each one an "HamiltonianOp."
 */
struct HamiltonianOp {
    bitLenInt targetBit;
    BitOp matrix;
    bitLenInt* controls;
    bitLenInt controlLen;

    HamiltonianOp(bitLenInt target, BitOp mtrx)
        : targetBit(target)
        , matrix(mtrx)
        , controls(NULL)
        , controlLen(0)
    {
    }

    HamiltonianOp(bitLenInt* controls, bitLenInt ctrlLen, bitLenInt target, BitOp mtrx)
        : targetBit(target)
        , matrix(mtrx)
        , controls(controls)
        , controlLen(ctrlLen)
    {
    }
};

/**
 * To define a Hamiltonian, give a vector of controlled single bit gates ("HamiltonianOp" instances) that are applied by
 * left-multiplication in low-to-high vector index order on the state vector.
 */
typedef std::shared_ptr<HamiltonianOp> HamiltonianOpPtr;
typedef std::vector<HamiltonianOpPtr> Hamiltonian;

} // namespace Qrack
