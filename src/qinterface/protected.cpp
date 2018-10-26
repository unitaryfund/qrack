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

#include <algorithm>

#include "qinterface.hpp"

namespace Qrack {

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride)
{
    while ((first < last) && (first < (last - stride))) {
        last -= stride;
        std::iter_swap(first, last);
        first += stride;
    }
}

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride)
{
    reverse(first, middle, stride);
    reverse(middle, last, stride);
    reverse(first, last, stride);
}

template void rotate<complex*>(complex* first, complex* middle, complex* last, bitCapInt stride);

bool does2x2PhaseShift(const complex* mtrx)
{
    bool doesShift = false;
    real1 phase = -M_PI * 2;
    for (int i = 0; i < 4; i++) {
        if (norm(mtrx[i]) > ZERO_R1) {
            if (phase < -M_PI) {
                phase = arg(mtrx[i]);
                continue;
            }

            real1 diff = arg(mtrx[i]) - phase;
            if (diff < ZERO_R1) {
                diff = -diff;
            }
            if (diff > M_PI) {
                diff = (2 * M_PI) - diff;
            }
            if (diff > min_norm) {
                doesShift = true;
                break;
            }
        }
    }
    return doesShift;
}

void QInterface::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubit;
    Apply2x2(0, qPowers[0], mtrx, 1, qPowers, doCalcNorm);
}

void QInterface::ApplyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen, const complex* mtrx, bitLenInt qubit)
{
    ApplyControlled2x2(controls, controlLen, qubit, mtrx, false);
}

void QInterface::ApplyControlled2x2(const bitLenInt* controls, const bitLenInt &controlLen, const bitLenInt &target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt* qPowers = new bitCapInt[controlLen + 1];
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 1];
    bitCapInt fullMask = 0;
    for (int i = 0; i < controlLen; i++) {
        qPowers[i] = 1 << controls[i];
	fullMask |= qPowers[i];
    }
    qPowers[controlLen] = 1 << target;
    fullMask |= qPowers[controlLen];
    std::copy(qPowers, qPowers + controlLen + 1, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 1);
    Apply2x2(qPowers[0], fullMask, mtrx, controlLen, qPowersSorted, doCalcNorm);
    delete[] qPowers;
    delete[] qPowersSorted;
}

void QInterface::ApplyControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << control;
    qPowers[1] = 1 << target;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowers[0], (qPowers[0]) | (qPowers[1]), mtrx, 2, qPowersSorted, doCalcNorm);
}

void QInterface::ApplyAntiControlled2x2(const bitLenInt* controls, const bitLenInt &controlLen, const bitLenInt &target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt* qPowers = new bitCapInt[controlLen + 1];
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 1];
    bitCapInt fullMask = 0;
    bitCapInt controlMask;
    for (int i = 0; i < controlLen; i++) {
        qPowers[i] = 1 << controls[i];
	fullMask |= qPowers[i];
    }
    controlMask = fullMask;
    qPowers[controlLen] = 1 << target;
    fullMask |= qPowers[controlLen];
    std::copy(qPowers, qPowers + controlLen + 1, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 1);
    Apply2x2(controlMask, fullMask, mtrx, controlLen, qPowersSorted, doCalcNorm);
    delete[] qPowers;
    delete[] qPowersSorted;
}

void QInterface::ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << control;
    qPowers[1] = 1 << target;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(0, qPowers[1], mtrx, 2, qPowersSorted, doCalcNorm);
}

void QInterface::ApplyDoublyControlled2x2(
    bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[3];
    qPowers[0] = 1 << control1;
    qPowers[1] = 1 << control2;
    qPowers[2] = 1 << target;
    std::copy(qPowers, qPowers + 3, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(
        (qPowers[0]) | (qPowers[1]), (qPowers[0]) | (qPowers[1]) | (qPowers[2]), mtrx, 3, qPowersSorted, doCalcNorm);
}

void QInterface::ApplyDoublyAntiControlled2x2(
    bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[3];
    qPowers[0] = 1 << control1;
    qPowers[1] = 1 << control2;
    qPowers[2] = 1 << target;
    std::copy(qPowers, qPowers + 3, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(0, qPowers[2], mtrx, 3, qPowersSorted, doCalcNorm);
}

} // namespace Qrack
