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

#pragma once

#include "common/qrack_types.hpp"

namespace Qrack {

class QParity;
typedef std::shared_ptr<QParity> QParityPtr;

class QParity {
public:
    /**
     * Measure (and collapse) parity of the masked set of qubits
     */
    virtual bool MParity(bitCapInt mask) { return ForceMParity(mask, false, false); }

    /**
     * If the target qubit set parity is odd, this applies a phase factor of \f$e^{i angle}\f$. If the target qubit set
     * parity is even, this applies the conjugate, e^{-i angle}.
     */
    virtual void UniformParityRZ(bitCapInt mask, real1_f angle)
    {
        CUniformParityRZ(std::vector<bitLenInt>(), mask, angle);
    }

    /**
     * Overall probability of any odd permutation of the masked set of bits
     */
    virtual real1_f ProbParity(bitCapInt mask) = 0;

    /**
     * Act as if is a measurement of parity of the masked set of qubits was applied, except force the (usually random)
     * result
     *
     * \warning PSEUDO-QUANTUM
     */
    virtual bool ForceMParity(bitCapInt mask, bool result, bool doForce = true) = 0;

    /**
     * If the controls are set and the target qubit set parity is odd, this applies a phase factor of \f$e^{i angle}\f$.
     * If the controls are set and the target qubit set parity is even, this applies the conjugate, \f$e^{-i angle}\f$.
     * Otherwise, do nothing if any control is not set.
     */
    virtual void CUniformParityRZ(const std::vector<bitLenInt>& controls, bitCapInt mask, real1_f angle) = 0;
};
} // namespace Qrack
