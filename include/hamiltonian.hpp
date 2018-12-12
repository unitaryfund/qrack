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
 * To define a Hamiltonian, give a vector of controlled single bit gates ("HamiltonianOp" instances) that are
 * applied by left-multiplication in low-to-high vector index order on the state vector.
 *
 * \warning Hamiltonian components might not commute, and observe the component factor of 2 * pi.
 *
 * As a general point of linear algebra, where A and B are linear operators, e^{i * (A + B) * t} = e^{i * A * t} *
 * e^{i * B * t} might NOT hold, if the operators A and B do not commute. As a rule of thumb, A will commute with B
 * at least in the case that A and B act on entirely different sets of qubits. However, for defining the intended
 * Hamiltonian, the programmer can be guaranteed that the exponential factors will be applied right-to-left, by left
 * multiplication, in the order e^(i * H_(N - 1) * t) * e^(i * H_(N - 2) * t) * ... e^(i * H_0 * t) * |psi>. (For
 * example, if A and B are single bit gates acting on the same bit, form their composition into one gate by the intended
 * right-to-left fusion and apply them as a single HamiltonianOp.)
 */
typedef std::shared_ptr<HamiltonianOp> HamiltonianOpPtr;
typedef std::vector<HamiltonianOpPtr> Hamiltonian;

} // namespace Qrack
