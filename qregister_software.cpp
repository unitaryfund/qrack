//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qregister.hpp"
#include <iostream>

#include "par_for.hpp"

namespace Qrack {

/// "Circular shift left" - shift bits left, and carry last bits.
void CoherentUnit::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt regMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt i;
    for (i = 0; i < length; i++) {
        regMask += 1 << (start + i);
    }
    otherMask -= regMask;
    bitCapInt bciArgs[6] = { regMask, otherMask, lengthPower - 1, start, shift, length };
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[1]));
            bitCapInt regRes = (lcv & (bciArgs[0]));
            bitCapInt regInt = regRes >> (bciArgs[3]);
            bitCapInt outInt = (regInt >> (bciArgs[5] - bciArgs[4])) | ((regInt << (bciArgs[4])) & bciArgs[2]);
            nStateVec[(outInt << (bciArgs[3])) + otherRes] = stateVec[lcv];
        });
    stateVec.reset();
    stateVec = std::move(nStateVec);
}

/// "Circular shift right" - shift bits right, and carry first bits.
void CoherentUnit::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt regMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt i;
    for (i = 0; i < length; i++) {
        regMask += 1 << (start + i);
    }
    otherMask -= regMask;
    bitCapInt bciArgs[6] = { regMask, otherMask, lengthPower - 1, start, shift, length };
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[1]));
            bitCapInt regRes = (lcv & (bciArgs[0]));
            bitCapInt regInt = regRes >> (bciArgs[3]);
            bitCapInt outInt = (regInt >> (bciArgs[4])) | ((regInt << (bciArgs[5] - bciArgs[4])) & bciArgs[2]);
            nStateVec[(outInt << (bciArgs[3])) + otherRes] = stateVec[lcv];
        });
    stateVec.reset();
    stateVec = std::move(nStateVec);
}

/// Add integer (without sign, with carry)
void CoherentUnit::INCC(bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[7] = { inOutMask, toAdd, carryMask, otherMask, lengthPower, inOutStart, carryIndex };
    par_for_skip(0, maxQPower, 1 << carryIndex, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[5]);
            bitCapInt outInt = inOutInt + bciArgs[1];
            bitCapInt outRes;
            if (outInt < (bciArgs[4])) {
                outRes = (outInt << (bciArgs[5])) | otherRes;
            } else {
                outRes = ((outInt - (bciArgs[4])) << (bciArgs[5])) | otherRes | (bciArgs[2]);
            }
            nStateVec[outRes] = stateVec[lcv];
        });
    ResetStateVec(std::move(nStateVec));
}

/// Add two quantum integers
/** Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
 * "inOutStart." */
/*void CoherentUnit::ADD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length)
{
    bitCapInt inOutMask = 0;
    bitCapInt inMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitLenInt i;
    for (i = 0; i < length; i++) {
        inOutMask += 1 << (inOutStart + i);
        inMask += 1 << (inStart + i);
    }
    otherMask -= inOutMask + inMask;
    bitCapInt bciArgs[6] = { inOutMask, inMask, otherMask, lengthPower - 1, inOutStart, inStart };
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[2]));
            if (otherRes == lcv) {
                nStateVec[lcv] = stateVec[lcv];
            } else {
                bitCapInt inOutRes = (lcv & (bciArgs[0]));
                bitCapInt inOutInt = inOutRes >> (bciArgs[4]);
                bitCapInt inRes = (lcv & (bciArgs[1]));
                bitCapInt inInt = inRes >> (bciArgs[5]);
                nStateVec[(((inOutInt + inInt) & bciArgs[3]) << (bciArgs[4])) | otherRes | inRes] = stateVec[lcv];
            }
        });
    stateVec.reset();
    stateVec = std::move(nStateVec);
}*/

/// Subtract two quantum integers
/** Subtract integer of "length" bits in "toSub" from integer of "length" bits in "inOutStart," and store result in
 * "inOutStart." */
/*void CoherentUnit::SUB(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length)
{
    bitCapInt inOutMask = 0;
    bitCapInt inMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitLenInt i;
    for (i = 0; i < length; i++) {
        inOutMask += 1 << (inOutStart + i);
        inMask += 1 << (toSub + i);
    }
    otherMask ^= inOutMask | inMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    bitCapInt bciArgs[6] = { inOutMask, inMask, otherMask, lengthPower - 1, inOutStart, toSub };
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[2]));
            if (otherRes == lcv) {
                nStateVec[lcv] = stateVec[lcv];
            } else {
                bitCapInt inOutRes = (lcv & (bciArgs[0]));
                bitCapInt inOutInt = inOutRes >> (bciArgs[4]);
                bitCapInt inRes = (lcv & (bciArgs[1]));
                bitCapInt inInt = inRes >> (bciArgs[5]);
                nStateVec[(((inOutInt - inInt + (bciArgs[3])) & bciArgs[3]) << (bciArgs[4])) + otherRes + inRes] =
                    stateVec[lcv];
            }
        });
    stateVec.reset();
    stateVec = std::move(nStateVec);
}*/

// Private CoherentUnit methods
void CoherentUnit::Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm)
{
    par_for(0, maxQPower, &(stateVec[0]), Complex16(doApplyNorm ? (1.0 / runningNorm) : 1.0, 0.0), mtrx, qPowersSorted,
        offset1, offset2, bitCount,
        [](const bitCapInt lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx,
            const bitCapInt offset1, const bitCapInt offset2) {
            Complex16 qubit[2];

            qubit[0] = stateVec[lcv + offset1];
            qubit[1] = stateVec[lcv + offset2];

            Complex16 Y0 = qubit[0];
            qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
            qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));

            stateVec[lcv + offset1] = qubit[0];
            stateVec[lcv + offset2] = qubit[1];
        });

    if (doCalcNorm) {
        UpdateRunningNorm();
    } else {
        runningNorm = 1.0;
    }
}
} // namespace Qrack
