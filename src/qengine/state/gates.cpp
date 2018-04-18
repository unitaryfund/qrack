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

#include "qengine_cpu.hpp"

namespace Qrack {

/// Measurement gate
bool QEngineCPU::M(bitLenInt qubit)
{
    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bool result;
    double prob = Rand();
    double angle = Rand() * 2.0 * M_PI;
    double cosine = cos(angle);
    double sine = sin(angle);
    Complex16 nrm;

    bitCapInt qPowers = 1 << qubit;
    double oneChance = Prob(qubit);

    result = (prob < oneChance) && oneChance > 0.0;
    double nrmlzr = 1.0;
    if (result) {
        if (oneChance > 0.0) {
            nrmlzr = oneChance;
        }

        nrm = Complex16(cosine, sine) / nrmlzr;

        par_for(0, maxQPower, [&](const bitCapInt lcv) {
            if ((lcv & qPowers) == 0) {
                stateVec[lcv] = Complex16(0.0, 0.0);
            } else {
                stateVec[lcv] = nrm * stateVec[lcv];
            }
        });
    } else {
        if (oneChance < 1.0) {
            nrmlzr = sqrt(1.0 - oneChance);
        }

        nrm = Complex16(cosine, sine) / nrmlzr;

        par_for(0, maxQPower, [&](const bitCapInt lcv) {
            if ((lcv & qPowers) == 0) {
                stateVec[lcv] = nrm * stateVec[lcv];
            } else {
                stateVec[lcv] = Complex16(0.0, 0.0);
            }
        });
    }

    UpdateRunningNorm();

    return result;
}

/// Bitwise swap
void QEngineCPU::Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        Swap(start1, start2);
        return;
    }

    int distance = start1 - start2;
    if (distance < 0) {
        distance *= -1;
    }
    if (distance < length) {
        bitLenInt i;
        for (i = 0; i < length; i++) {
            Swap(start1 + i, start2 + i);
        }
    } else {
        bitCapInt reg1Mask = ((1 << length) - 1) << start1;
        bitCapInt reg2Mask = ((1 << length) - 1) << start2;
        bitCapInt otherMask = maxQPower - 1;
        otherMask ^= reg1Mask | reg2Mask;
        Complex16 *nStateVec = new Complex16[maxQPower];

        par_for(0, maxQPower, [&](const bitCapInt lcv) {
            bitCapInt otherRes = (lcv & otherMask);
            bitCapInt reg1Res = ((lcv & reg1Mask) >> (start1)) << (start2);
            bitCapInt reg2Res = ((lcv & reg2Mask) >> (start2)) << (start1);
            nStateVec[reg1Res | reg2Res | otherRes] = stateVec[lcv];
        });
        // We replace our old permutation state vector with the new one we just filled, at the end.
        ResetStateVec(nStateVec);
    }
}

/// Phase flip always - equivalent to Z X Z X on any bit in the QEngineCPU
void QEngineCPU::PhaseFlip()
{
    par_for(0, maxQPower, [&](const bitCapInt lcv) { stateVec[lcv] = -stateVec[lcv]; });
}

}
