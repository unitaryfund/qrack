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

namespace Qrack {

/**
 * A Hamiltonian can be specified in terms of arbitrary controlled single bit gates, each one an "HamiltonianOp."
 */
struct HamiltonianOp {
    bitLenInt targetBit;
    BitOp matrix;
    std::vector<bitLenInt> controls;

    HamiltonianOp(bitLenInt target, BitOp mtrx)
        : targetBit(target)
        , matrix(mtrx)
        , controls(0)
    {
    }

    HamiltonianOp(std::vector<bitLenInt> controls, bitLenInt target, BitOp mtrx)
        : targetBit(target)
        , matrix(mtrx)
        , controls(controls)
    {
    }
}

/**
 * To define a Hamiltonian, give a vector of controlled single bit gates ("HamiltonianOp" instances) that are applied by
 * left-multiplication in low-to-high vector index order on the state vector.
 */
typedef std::vector<HamiltonianOp>
    Hamiltonian;

} // namespace Qrack
