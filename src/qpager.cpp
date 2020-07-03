//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// When we allocate a quantum register, all bits are in a (re)set state. At this point,
// we know they are separable, in the sense of full Schmidt decomposability into qubits
// in the "natural" or "permutation" basis of the register. Many operations can be
// traced in terms of fewer qubits that the full "Schr\{"o}dinger representation."
//
// Based on experimentation, QUnit is designed to avoid increasing representational
// entanglement for its primary action, and only try to decrease it when inquiries
// about probability need to be made otherwise anyway. Avoiding introducing the cost of
// basically any entanglement whatsoever, rather than exponentially costly "garbage
// collection," should be the first and ultimate concern, in the authors' experience.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qpager.hpp"

namespace Qrack {

QPager::QPager(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool ignored, bool ignored2, bool useHostMem, int deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1 norm_thresh, std::vector<bitLenInt> devList)
    : QInterface(qBitCount, rgp, ignored, useHardwareRNG, false, norm_thresh)
    , engine(eng)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
{
    SetQubitCount(qubitCount);

    if (pow2(qubitsPerPage) > (sizeof(bitCapIntOcl) * bitsInByte)) {
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");
    }

    bool isPermInPage;
    bitCapInt pagePerm = 0;
    for (bitCapInt i = 0; i < qPageCount; i++) {
        isPermInPage = (initState >= pagePerm);
        pagePerm += qPageMaxQPower;
        isPermInPage &= (initState < pagePerm);
        if (isPermInPage) {
            qPages.push_back(MakeEngine(qPageQubitCount, initState - (pagePerm - qPageMaxQPower)));
        } else {
            qPages.push_back(MakeEngine(qPageQubitCount, 0));
            qPages.back()->SetAmplitude(0, ZERO_CMPLX);
        }
    }
}

void QPager::CombineEngines()
{
    if (qPages.size() == 1U) {
        return;
    }

    std::vector<QEnginePtr> nQPages;
    nQPages.push_back(MakeEngine(qubitCount, 0));
    for (bitCapInt i = 0; i < qPageCount; i++) {
        nQPages[0]->SetAmplitudePage(qPages[i], 0, i * qPageMaxQPower, qPageMaxQPower);
    }

    qPages = nQPages;
}

void QPager::SeparateEngines()
{
    if (qPages.size() == qPageCount) {
        return;
    }

    std::vector<QEnginePtr> nQPages;
    for (bitCapInt i = 0; i < qPageCount; i++) {
        nQPages.push_back(MakeEngine(qPageQubitCount, 0));
        nQPages.back()->SetAmplitudePage(qPages[0], i * qPageMaxQPower, 0, qPageMaxQPower);
    }

    qPages = nQPages;
}

bitLenInt QPager::Compose(QPagerPtr toCopy)
{
    CombineEngines();
    toCopy->CombineEngines();
    bitLenInt toRet = qPages[0]->Compose(toCopy->qPages[0]);
    SetQubitCount(qPages[0]->GetQubitCount());
    toCopy->SeparateEngines();
    SeparateEngines();
    return toRet;
}

bitLenInt QPager::Compose(QPagerPtr toCopy, bitLenInt start)
{
    CombineEngines();
    toCopy->CombineEngines();
    bitLenInt toRet = qPages[0]->Compose(toCopy->qPages[0], start);
    SetQubitCount(qPages[0]->GetQubitCount());
    toCopy->SeparateEngines();
    SeparateEngines();
    return toRet;
}

void QPager::Decompose(bitLenInt start, bitLenInt length, QPagerPtr dest)
{
    CombineEngines();
    dest->CombineEngines();
    qPages[0]->Decompose(start, length, dest->qPages[0]);
    SetQubitCount(qPages[0]->GetQubitCount());
    dest->SeparateEngines();
    SeparateEngines();
}

void QPager::Dispose(bitLenInt start, bitLenInt length)
{
    CombineEngines();
    qPages[0]->Dispose(start, length);
    SeparateEngines();
}

void QPager::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    CombineEngines();
    qPages[0]->Dispose(start, length, disposedPerm);
    SeparateEngines();
}

void QPager::SetQuantumState(const complex* inputState)
{
    bitCapInt pagePerm = 0;
    for (bitCapInt i = 0; i < qPageCount; i++) {
        qPages[i]->SetQuantumState(inputState + pagePerm);
        pagePerm += qPageMaxQPower;
    }
}

void QPager::GetQuantumState(complex* outputState)
{
    bitCapInt pagePerm = 0;
    for (bitCapInt i = 0; i < qPageCount; i++) {
        qPages[i]->GetQuantumState(outputState + pagePerm);
        pagePerm += qPageMaxQPower;
    }
}

void QPager::GetProbs(real1* outputProbs)
{
    bitCapInt pagePerm = 0;
    for (bitCapInt i = 0; i < qPageCount; i++) {
        qPages[i]->GetProbs(outputProbs + pagePerm);
        pagePerm += qPageMaxQPower;
    }
}

void QPager::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool isPermInPage;
    bitCapInt pagePerm = 0;
    for (bitCapInt i = 0; i < qPageCount; i++) {
        isPermInPage = (perm >= pagePerm);
        pagePerm += qPageMaxQPower;
        isPermInPage &= (perm < pagePerm);

        if (isPermInPage) {
            qPages[i]->SetPermutation(perm - (pagePerm - qPageMaxQPower));
            continue;
        }

        qPages[i]->ZeroAmplitudes();
    }
}

void QPager::ApplySingleBit(const complex* mtrx, bitLenInt target)
{
    if (IsIdentity(mtrx, true)) {
        return;
    }

    if ((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) {
        ApplySinglePhase(mtrx[0], mtrx[3], target);
        return;
    }

    if ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)) {
        ApplySingleInvert(mtrx[1], mtrx[2], target);
        return;
    }

    if (target < qubitsPerPage) {
        for (bitCapInt i = 0; i < qPageCount; i++) {
            qPages[i]->ApplySingleBit(mtrx, target);
        }
        return;
    }

    // Here, the gate requires data to cross sub-engine boundaries.
    // It's always a matter of swapping the high halves of half the sub-engines with the low halves of the other
    // half of engines, acting the maximum bit gate, (for the sub-engine bit count,) and swapping back. Depending on
    // the bit index and number of sub-engines, we just have to determine which sub-engine to pair with which.
    bitCapInt groupCount = ONE_BCI << (qubitCount - (target + ONE_BCI));
    bitCapInt groupSize = ONE_BCI << ((target + ONE_BCI) - qubitsPerPage);
    bitLenInt sqi = qubitsPerPage - ONE_BCI;

    bitCapInt i, j;

    for (i = 0; i < groupCount; i++) {
        for (j = 0; j < (groupSize / 2); j++) {
            QEnginePtr engine1 = qPages[j + (i * groupSize)];
            QEnginePtr engine2 = qPages[j + (i * groupSize) + (groupSize / 2)];

            engine1->ShuffleBuffers(engine2);

            engine1->ApplySingleBit(mtrx, sqi);
            engine2->ApplySingleBit(mtrx, sqi);

            engine1->ShuffleBuffers(engine2);
        }
    }
}

void QPager::ApplySinglePhase(const complex tl, const complex br, bitLenInt target)
{
    complex topLeft = tl;
    complex bottomRight = br;

    if ((topLeft == bottomRight) && (randGlobalPhase || (topLeft == ONE_CMPLX))) {
        return;
    }

    if (target < qubitsPerPage) {
        for (bitCapInt i = 0; i < qPageCount; i++) {
            qPages[i]->ApplySinglePhase(topLeft, bottomRight, target);
        }
        return;
    }

    if (randGlobalPhase) {
        topLeft = ONE_CMPLX;
        bottomRight /= topLeft;
    }

    bitCapInt offset = pow2(target - qubitsPerPage);
    bitCapInt qMask = offset - ONE_BCI;
    bitCapInt maxLcv = qPageCount >> ONE_BCI;
    bitCapInt i;
    for (bitCapInt lcv = 0; lcv < maxLcv; lcv++) {
        i = lcv & qMask;
        i |= (lcv ^ i) << ONE_BCI;

        if (topLeft != ONE_CMPLX) {
            qPages[i]->ApplySinglePhase(topLeft, topLeft, 0);
        }

        if (bottomRight != ONE_CMPLX) {
            qPages[i + offset]->ApplySinglePhase(bottomRight, bottomRight, 0);
        }
    }
}

void QPager::ApplySingleInvert(const complex tr, const complex bl, bitLenInt target)
{
    complex topRight = tr;
    complex bottomLeft = bl;

    if ((topRight == -bottomLeft) && (randGlobalPhase || (topRight == ONE_CMPLX))) {
        return;
    }

    if (target < qubitsPerPage) {
        for (bitCapInt i = 0; i < qPageCount; i++) {
            qPages[i]->ApplySingleInvert(topRight, bottomLeft, target);
        }
        return;
    }

    if (randGlobalPhase) {
        topRight = ONE_CMPLX;
        bottomLeft /= topRight;
    }

    bitCapInt offset = pow2(target - qubitsPerPage);
    bitCapInt qMask = offset - ONE_BCI;
    bitCapInt maxLcv = qPageCount >> ONE_BCI;
    bitCapInt i;
    for (bitCapInt lcv = 0; lcv < maxLcv; lcv++) {
        i = lcv & qMask;
        i |= (lcv ^ i) << ONE_BCI;

        std::swap(qPages[i], qPages[i + offset]);

        if (topRight != ONE_CMPLX) {
            qPages[i]->ApplySinglePhase(topRight, topRight, 0);
        }

        if (bottomLeft != ONE_CMPLX) {
            qPages[i + offset]->ApplySinglePhase(bottomLeft, bottomLeft, 0);
        }
    }
}

} // namespace Qrack
