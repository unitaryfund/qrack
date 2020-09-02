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

#define CHECK_ZERO_SKIP()                                                                                              \
    if (!stateVec) {                                                                                                   \
        return;                                                                                                        \
    }

namespace Qrack {

void QEngineCPU::ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
{
    CHECK_ZERO_SKIP();

    dispatchQueue.restart();

    ParallelFunc fn = [&](const bitCapInt i, const int cpu) {
        if ((i & regMask) == result) {
            stateVec->write(i, nrm * stateVec->read(i));
        } else {
            stateVec->write(i, complex(ZERO_R1, ZERO_R1));
        }
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPower, fn);
    }

    runningNorm = ONE_R1;
}

/// Phase flip always - equivalent to Z X Z X on any bit in the QEngineCPU
void QEngineCPU::PhaseFlip()
{
    CHECK_ZERO_SKIP();

    // This gate has no physical consequence. We only enable it for "book-keeping," if the engine is not using global
    // phase offsets.
    if (randGlobalPhase) {
        return;
    }

    dispatchQueue.restart();

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) { stateVec->write(lcv, -stateVec->read(lcv)); };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPower, fn);
    }
}

} // namespace Qrack
