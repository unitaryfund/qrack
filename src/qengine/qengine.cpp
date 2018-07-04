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

template void rotate<complex*>(complex* first, complex* middle, complex* last, bitCapInt stride);

void QEngineCPU::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit)
{
    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubit;
    Apply2x2(0, qPowers[0], mtrx, 1, qPowers, doCalcNorm);
}

void QEngineCPU::ApplyControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << control;
    qPowers[1] = 1 << target;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowers[0], qPowers[0] + qPowers[1], mtrx, 2, qPowersSorted, doCalcNorm);
}

void QEngineCPU::ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << control;
    qPowers[1] = 1 << target;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(0, qPowers[1], mtrx, 2, qPowersSorted, doCalcNorm);
}

void QEngineCPU::ApplyDoublyControlled2x2(
    bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[3];
    qPowers[0] = 1 << control1;
    qPowers[1] = 1 << control2;
    qPowers[2] = 1 << target;
    std::copy(qPowers, qPowers + 3, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(qPowers[0] + qPowers[1], qPowers[0] + qPowers[1] + qPowers[2], mtrx, 3, qPowersSorted, doCalcNorm);
}

void QEngineCPU::ApplyDoublyAntiControlled2x2(
    bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
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
