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

template void rotate<Complex16 *>(Complex16 *first, Complex16 *middle, Complex16 *last, bitCapInt stride);

/// Swap values of two bits in register
void QEngineCPU::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    // Does not necessarily commute with single bit gates
    FlushQueue(qubit1);
    FlushQueue(qubit2);

    if (qubit1 != qubit2) {
        const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0),
            Complex16(0.0, 0.0) };

        bitCapInt qPowers[3];
        bitCapInt qPowersSorted[2];
        qPowers[1] = 1 << qubit1;
        qPowers[2] = 1 << qubit2;
        qPowers[0] = qPowers[1] + qPowers[2];
        if (qubit1 < qubit2) {
            qPowersSorted[0] = qPowers[1];
            qPowersSorted[1] = qPowers[2];
        } else {
            qPowersSorted[0] = qPowers[2];
            qPowersSorted[1] = qPowers[1];
        }

        Apply2x2(qPowers[2], qPowers[1], pauliX, 2, qPowersSorted, false);
    }
}

void QEngineCPU::ApplySingleBit(bitLenInt qubit, const Complex16* mtrx, bool doCalcNorm)
{
    if (qubitCount <= 3) {
        bitCapInt qPowers[1];
        qPowers[0] = 1 << qubit;
        Apply2x2(0, qPowers[0], mtrx, 1, qPowers, doCalcNorm);
    }
    else if (isQueued[qubit]) {
        Mul2x2(mtrx, &(gateQueue[qubit][0]));
    }
    else {
        isQueued[qubit] = true;
        std::copy(mtrx, mtrx + 4, &(gateQueue[qubit][0]));
    }
}

void QEngineCPU::ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm)
{
    // Does not necessarily commute with single bit gates
    FlushQueue(control);
    FlushQueue(target);

    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[2];
    qPowers[1] = 1 << control;
    qPowers[2] = 1 << target;
    qPowers[0] = qPowers[1] + qPowers[2];
    if (control < target) {
        qPowersSorted[0] = qPowers[1];
        qPowersSorted[1] = qPowers[2];
    } else {
        qPowersSorted[0] = qPowers[2];
        qPowersSorted[1] = qPowers[1];
    }
    Apply2x2(qPowers[0], qPowers[1], mtrx, 2, qPowersSorted, doCalcNorm);
}

void QEngineCPU::ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm)
{
    // Does not necessarily commute with single bit gates
    FlushQueue(control);
    FlushQueue(target);

    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[2];
    qPowers[1] = 1 << control;
    qPowers[2] = 1 << target;
    qPowers[0] = qPowers[1] + qPowers[2];
    if (control < target) {
        qPowersSorted[0] = qPowers[1];
        qPowersSorted[1] = qPowers[2];
    } else {
        qPowersSorted[0] = qPowers[2];
        qPowersSorted[1] = qPowers[1];
    }
    Apply2x2(0, qPowers[2], mtrx, 2, qPowersSorted, doCalcNorm);
}

void QEngineCPU::Reverse(bitLenInt first, bitLenInt last)
{
    while ((first < last) && (first < (last - 1))) {
        last--;
        Swap(first, last);
        first++;
    }
}

void QEngineCPU::Mul2x2(const Complex16* leftIn, Complex16* rightOut) {
    Complex16 rightDupe[4];
    std::copy(rightOut, rightOut + 4, rightDupe);
    std::vector<std::future<void>> futures(4);
    futures[0] = std::async(std::launch::async, [rightOut, leftIn, rightDupe]() {
        rightOut[0] = leftIn[0] * rightDupe[0] + leftIn[1] * rightDupe[2];
    });
    futures[1] = std::async(std::launch::async, [rightOut, leftIn, rightDupe]() {
        rightOut[1] = leftIn[0] * rightDupe[1] + leftIn[1] * rightDupe[3];
    });
    futures[2] = std::async(std::launch::async, [rightOut, leftIn, rightDupe]() {
        rightOut[2] = leftIn[2] * rightDupe[0] + leftIn[3] * rightDupe[2];
    });
    futures[3] = std::async(std::launch::async, [rightOut, leftIn, rightDupe]() {
        rightOut[3] = leftIn[2] * rightDupe[1] + leftIn[3] * rightDupe[3];
    });
    for (int i = 0; i < 4; i++) {
        futures[i].get();
    }
}

void QEngineCPU::FlushQueue(bitLenInt index) {
    if (isQueued[index]) {
        isQueued[index] = false;
        bitCapInt qPowers[1];
        qPowers[0] = 1 << index;
        Apply2x2(0, qPowers[0], &(gateQueue[index][0]), 1, qPowers, true);
    }
}

void QEngineCPU::FlushQueue(bitLenInt start, bitLenInt length) {
    for (bitLenInt i = 0; i < length; i++) {
        FlushQueue(start + i);
    }
}

bool QEngineCPU::CheckQueued(bitLenInt start, bitLenInt length) {
    for (bitLenInt i = 0; i < length; i++) {
        if (isQueued[start + i]) {
            return true;
        }
    }
    return false;
}

} // namespace Qrack
