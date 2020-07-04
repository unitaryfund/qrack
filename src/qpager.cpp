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

#include <future>

#include "qfactory.hpp"

namespace Qrack {

QPager::QPager(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool ignored, bool ignored2, bool useHostMem, int deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1 norm_thresh, std::vector<bitLenInt> devList)
    : QInterface(qBitCount, rgp, false, useHardwareRNG, false, norm_thresh)
    , engine(eng)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
{
    if ((eng != QINTERFACE_CPU) && (eng != QINTERFACE_OPENCL)) {
        throw std::invalid_argument("QPager sub-engine type must be QINTERFACE_CPU or QINTERFACE_OPENCL.");
    }

    SetQubitCount(qubitCount);

    if (qubitsPerPage > (sizeof(bitCapIntOcl) * bitsInByte)) {
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
            qPages.push_back(MakeEngine(qubitsPerPage, initState - (pagePerm - qPageMaxQPower)));
        } else {
            qPages.push_back(MakeEngine(qubitsPerPage, 0));
            qPages.back()->SetAmplitude(0, ZERO_CMPLX);
        }
    }
}

QEnginePtr QPager::MakeEngine(bitLenInt length, bitCapInt perm)
{
    return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engine, length, perm, rand_generator, phaseFactor,
        false, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse));
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
        nQPages.push_back(MakeEngine(qubitsPerPage, 0));
        nQPages.back()->SetAmplitudePage(qPages[0], i * qPageMaxQPower, 0, qPageMaxQPower);
    }

    qPages = nQPages;
}

// This is like the QEngineCPU and QEngineOCL logic for register-like CNOT and CCNOT, just swapping sub-engine indices
// instead of amplitude indices.
void QPager::MetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target,
    std::vector<bitLenInt> intraControls, const complex* mtrx)
{
    bitCapInt i;

    bitLenInt maxLcv = qPageCount >> (1U + controls.size());
    std::vector<bitLenInt> sortedMasks(1U + controls.size());
    sortedMasks[controls.size()] = 1U << target;

    bitCapInt controlMask = 0;
    for (i = 0; i < controls.size(); i++) {
        sortedMasks[i] = 1U << controls[i];
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapInt targetMask = pow2(target - qPagePow);
    bitLenInt sqi = qubitsPerPage - 1U;

    for (i = 0; i < maxLcv; i++) {
        bitCapInt j, k, jLo, jHi;
        jHi = i;
        j = 0;
        for (k = 0; k < (sortedMasks.size()); k++) {
            jLo = jHi & sortedMasks[k];
            jHi = (jHi ^ jLo) << ONE_BCI;
            j |= jLo;
        }
        j |= jHi | controlMask;

        QEnginePtr engine1 = qPages[j];
        QEnginePtr engine2 = qPages[j + targetMask];

        engine1->ShuffleBuffers(engine2);

        if (intraControls.size() == 0) {
            engine1->ApplySingleBit(mtrx, sqi);
            engine2->ApplySingleBit(mtrx, sqi);
        } else if (anti) {
            engine1->ApplyAntiControlledSingleBit(&(intraControls[0]), intraControls.size(), sqi, mtrx);
            engine2->ApplyAntiControlledSingleBit(&(intraControls[0]), intraControls.size(), sqi, mtrx);
        } else {
            engine1->ApplyControlledSingleBit(&(intraControls[0]), intraControls.size(), sqi, mtrx);
            engine2->ApplyControlledSingleBit(&(intraControls[0]), intraControls.size(), sqi, mtrx);
        }

        engine1->ShuffleBuffers(engine2);
    }
}

// This is called when control bits are "meta-" but the target bit is below the "meta-" threshold, (low enough to fit in
// sub-engines).
void QPager::SemiMetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt targetBit,
    std::vector<bitLenInt> intraControls, const complex* mtrx)
{
    bitCapInt i;
    bitCapInt maxLcv = qPageCount >> (controls.size());
    std::vector<bitLenInt> sortedMasks(controls.size());

    bitCapInt controlMask = 0;
    for (i = 0; i < controls.size(); i++) {
        sortedMasks[i] = 1U << controls[i];
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapInt j, k, jLo, jHi;
    for (i = 0; i < maxLcv; i++) {
        jHi = i;
        j = 0;
        for (k = 0; k < (sortedMasks.size()); k++) {
            jLo = jHi & sortedMasks[k];
            jHi = (jHi ^ jLo) << ONE_BCI;
            j |= jLo;
        }
        j |= jHi | controlMask;

        if (intraControls.size() == 0) {
            qPages[j]->ApplySingleBit(mtrx, targetBit);
        } else if (anti) {
            qPages[j]->ApplyAntiControlledSingleBit(&(intraControls[0]), intraControls.size(), targetBit, mtrx);
        } else {
            qPages[j]->ApplyControlledSingleBit(&(intraControls[0]), intraControls.size(), targetBit, mtrx);
        }
    }
}

void QPager::MetaControlledPhaseInvert(bool anti, bool invert, std::vector<bitLenInt> controls, bitLenInt target,
    std::vector<bitLenInt> intraControls, complex top, complex bottom)
{
    bitCapInt i;

    std::vector<bitLenInt> sortedMasks(1U + controls.size());
    sortedMasks[controls.size()] = 1U << target;

    bitCapInt controlMask = 0;
    for (i = 0; i < controls.size(); i++) {
        sortedMasks[i] = 1U << controls[i];
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }

    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapInt targetMask = pow2(target - qPagePow);

    bitLenInt maxLcv = qPageCount >> (sortedMasks.size());

    for (i = 0; i < maxLcv; i++) {
        bitCapInt j, k, jLo, jHi;
        jHi = i;
        j = 0;
        for (k = 0; k < (sortedMasks.size()); k++) {
            jLo = jHi & sortedMasks[k];
            jHi = (jHi ^ jLo) << ONE_BCI;
            j |= jLo;
        }
        j |= jHi | controlMask;

        if (invert) {
            std::swap(qPages[j], qPages[j + targetMask]);
        }

        QEnginePtr engine1 = qPages[j];
        QEnginePtr engine2 = qPages[j + targetMask];

        if (top != ONE_CMPLX) {
            if (intraControls.size() == 0) {
                engine1->ApplySinglePhase(top, top, 0);
            } else if (anti) {
                engine1->ApplyAntiControlledSinglePhase(&(intraControls[0]), intraControls.size(), 0, top, top);
            } else {
                engine1->ApplyControlledSinglePhase(&(intraControls[0]), intraControls.size(), 0, top, top);
            }
        }

        if (bottom != ONE_CMPLX) {
            if (intraControls.size() == 0) {
                engine2->ApplySinglePhase(bottom, bottom, 0);
            } else if (anti) {
                engine2->ApplyAntiControlledSinglePhase(&(intraControls[0]), intraControls.size(), 0, bottom, bottom);
            } else {
                engine2->ApplyControlledSinglePhase(&(intraControls[0]), intraControls.size(), 0, bottom, bottom);
            }
        }
    }
}

template <typename F> void QPager::CombineAndOp(F fn, std::vector<bitLenInt> bits)
{
    if (qPageCount == 1U) {
        fn(qPages[0]);
        return;
    }

    bitLenInt i;
    bitLenInt highestBit = 0;
    for (i = 0; i < bits.size(); i++) {
        if (bits[i] > highestBit) {
            highestBit = bits[i];
        }
    }

    if (highestBit >= qubitsPerPage) {
        CombineEngines();
    }

    fn(qPages[0]);

    if (highestBit >= qubitsPerPage) {
        SeparateEngines();
    }
}

template <typename F>
void QPager::CombineAndOpControlled(
    F fn, std::vector<bitLenInt> bits, const bitLenInt* controls, const bitLenInt controlLen)
{
    for (bitLenInt i = 0; i < controlLen; i++) {
        bits.push_back(controls[i]);
    }

    CombineAndOp(fn, bits);
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

void QPager::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    std::vector<bitLenInt> metaControls;
    std::vector<bitLenInt> intraControls;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (controls[i] < qubitsPerPage) {
            intraControls.push_back(controls[i]);
        } else {
            metaControls.push_back(controls[i]);
        }
    }

    if (target < qubitsPerPage) {
        SemiMetaControlled(false, metaControls, target, intraControls, mtrx);
    } else {
        MetaControlled(false, metaControls, target, intraControls, mtrx);
    }
}

void QPager::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    std::vector<bitLenInt> metaControls;
    std::vector<bitLenInt> intraControls;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (controls[i] < qubitsPerPage) {
            intraControls.push_back(controls[i]);
        } else {
            metaControls.push_back(controls[i]);
        }
    }

    if (target < qubitsPerPage) {
        SemiMetaControlled(true, metaControls, target, intraControls, mtrx);
    } else {
        MetaControlled(true, metaControls, target, intraControls, mtrx);
    }
}

void QPager::ApplyControlledPhaseInvert(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex top, const complex bottom, const bool isAnti, const bool isInvert)
{
    std::vector<bitLenInt> metaControls;
    std::vector<bitLenInt> intraControls;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (controls[i] < qubitsPerPage) {
            intraControls.push_back(controls[i]);
        } else {
            metaControls.push_back(controls[i]);
        }
    }

    if (target < qubitsPerPage) {
        complex mtrx[4];
        if (isInvert) {
            mtrx[0] = ZERO_CMPLX;
            mtrx[1] = top;
            mtrx[2] = bottom;
            mtrx[3] = ZERO_CMPLX;
        } else {
            mtrx[0] = top;
            mtrx[1] = ZERO_CMPLX;
            mtrx[2] = ZERO_CMPLX;
            mtrx[3] = bottom;
        }
        SemiMetaControlled(isAnti, metaControls, target, intraControls, mtrx);
    } else {
        MetaControlledPhaseInvert(isAnti, isInvert, metaControls, target, intraControls, top, bottom);
    }
}

void QPager::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) {
            engine->UniformlyControlledSingleBit(
                controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
        },
        { qubitIndex }, controls, controlLen);
}

void QPager::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CSqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CISqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}

bool QPager::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    CombineEngines();
    bool toRet = qPages[0]->ForceM(qubit, result, doForce, doApply);
    SeparateEngines();
    return toRet;
}

void QPager::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    CombineAndOp(
        [&](QEnginePtr engine) { engine->INC(toAdd, start, length); }, { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::CINC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CINC(toAdd, start, length, controls, controlLen); },
        { static_cast<bitLenInt>(start + length - 1U) }, controls, controlLen);
}
void QPager::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCS(toAdd, start, length, overflowIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex });
}
void QPager::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCSC(toAdd, start, length, overflowIndex, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex, carryIndex });
}
void QPager::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCSC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCBCD(toAdd, start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCBCDC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECSC(toSub, start, length, overflowIndex, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex, carryIndex });
}
void QPager::DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECSC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECBCDC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
void QPager::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->MUL(toMul, inOutStart, carryStart, length); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) });
}
void QPager::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DIV(toDiv, inOutStart, carryStart, length); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) });
}
void QPager::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->MULModNOut(toMul, modN, inStart, outStart, length); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) });
}
void QPager::IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->IMULModNOut(toMul, modN, inStart, outStart, length); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) });
}
void QPager::POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->POWModNOut(base, modN, inStart, outStart, length); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) });
}
void QPager::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) },
        controls, controlLen);
}
void QPager::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) },
        controls, controlLen);
}
void QPager::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
}
void QPager::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
}
void QPager::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
}

void QPager::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->ZeroPhaseFlip(start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex); },
        { static_cast<bitLenInt>(start + length - 1U), flagIndex });
}
void QPager::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->PhaseFlipIfLess(greaterPerm, start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::PhaseFlip()
{
    for (bitLenInt i = 0; i < qPageCount; i++) {
        qPages[i]->PhaseFlip();
    }
}

bitCapInt QPager::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    unsigned char* values, bool resetValue)
{
    CombineAndOp(
        [&](QEnginePtr engine) { engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, true); },
        { static_cast<bitLenInt>(indexStart + indexLength - 1U),
            static_cast<bitLenInt>(valueStart + valueLength - 1U) });

    return 0;
}

bitCapInt QPager::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    CombineAndOp(
        [&](QEnginePtr engine) {
            engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        },
        { static_cast<bitLenInt>(indexStart + indexLength - 1U), static_cast<bitLenInt>(valueStart + valueLength - 1U),
            carryIndex });

    return 0;
}
bitCapInt QPager::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, unsigned char* values)
{
    CombineAndOp(
        [&](QEnginePtr engine) {
            engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        },
        { static_cast<bitLenInt>(indexStart + indexLength - 1U), static_cast<bitLenInt>(valueStart + valueLength - 1U),
            carryIndex });

    return 0;
}
void QPager::Hash(bitLenInt start, bitLenInt length, unsigned char* values)
{
    CombineAndOp([&](QEnginePtr engine) { engine->Hash(start, length, values); },
        { static_cast<bitLenInt>(start + length - 1U) });
}

void QPager::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    CombineAndOp([&](QEnginePtr engine) { engine->Swap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::ISwap(bitLenInt qubit1, bitLenInt qubit2)
{
    CombineAndOp([&](QEnginePtr engine) { engine->ISwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    CombineAndOp([&](QEnginePtr engine) { engine->SqrtSwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    CombineAndOp([&](QEnginePtr engine) { engine->ISqrtSwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::FSim(real1 theta, real1 phi, bitLenInt qubit1, bitLenInt qubit2)
{
    CombineAndOp([&](QEnginePtr engine) { engine->FSim(theta, phi, qubit1, qubit2); }, { qubit1, qubit2 });
}

real1 QPager::Prob(bitLenInt qubitIndex)
{
    if (qPageCount == 1U) {
        return qPages[0]->Prob(qubitIndex);
    }

    real1 oneChance = ZERO_R1;
    bitCapInt i;

    if (qubitIndex < qubitsPerPage) {
        std::vector<std::future<real1>> futures(qPageCount);
        for (i = 0; i < qPageCount; i++) {
            QEnginePtr engine = qPages[i];
            futures[i] = std::async(std::launch::async, [engine, qubitIndex]() { return engine->Prob(qubitIndex); });
        }
        for (i = 0; i < qPageCount; i++) {
            oneChance += futures[i].get();
        }
    } else {
        CombineAndOp([&](QEnginePtr engine) { oneChance = engine->Prob(qubitIndex); }, { qubitIndex });
    }

    return oneChance;
}
real1 QPager::ProbAll(bitCapInt fullRegister)
{
    bitCapInt subIndex = fullRegister / qPageMaxQPower;
    fullRegister -= subIndex * qPageMaxQPower;
    return qPages[subIndex]->ProbAll(fullRegister);
}

bool QPager::ApproxCompare(QInterfacePtr toCompare)
{
    QPagerPtr toComparePager = std::dynamic_pointer_cast<QPager>(toCompare);
    CombineEngines();
    toComparePager->CombineEngines();
    bool toRet = qPages[0]->ApproxCompare(toComparePager->qPages[0]);
    toComparePager->SeparateEngines();
    SeparateEngines();
    return toRet;
}
void QPager::UpdateRunningNorm(real1 norm_thresh)
{
    for (bitCapInt i = 0; i < qPageCount; i++) {
        qPages[i]->UpdateRunningNorm(norm_thresh);
    }
}

QInterfacePtr QPager::Clone()
{
    QPagerPtr clone = std::dynamic_pointer_cast<QPager>(
        CreateQuantumInterface(QINTERFACE_QPAGER, engine, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
            randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse));
    for (bitCapInt i = 0; i < qPageCount; i++) {
        clone->qPages[i]->SetAmplitudePage(qPages[i], 0, 0, qPageMaxQPower);
    }
    return clone;
}

} // namespace Qrack
