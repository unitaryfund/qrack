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

#include "qengine_cpu.hpp"

namespace Qrack {

void QEngineCPU::ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
{
    par_for(0, maxQPower, [&](const bitCapInt i, const int cpu) {
        if ((i & regMask) == result) {
            stateVec->set(i, nrm * stateVec->get(i));
        } else {
            stateVec->set(i, complex(ZERO_R1, ZERO_R1));
        }
    });

    UpdateRunningNorm();
}

/// Phase flip always - equivalent to Z X Z X on any bit in the QEngineCPU
void QEngineCPU::PhaseFlip()
{
    // This gate has no physical consequence. We only enable it for "book-keeping," if the engine is not using global
    // phase offsets.
    if (!randGlobalPhase) {
        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) { stateVec->set(lcv, -stateVec->get(lcv)); });
    }
}

} // namespace Qrack
