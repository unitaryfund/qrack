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

void QPager::CombineEngines(bitLenInt ignored)
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

template <typename Qubit1Fn> void QPager::SingleBitGate(bitLenInt target, Qubit1Fn fn)
{
    if (target < qubitsPerPage) {
        for (bitCapInt i = 0; i < qPageCount; i++) {
            fn(qPages[i], target);
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

            std::future<void> future1 = std::async(std::launch::async, [engine1, fn, sqi]() { fn(engine1, sqi); });
            std::future<void> future2 = std::async(std::launch::async, [engine2, fn, sqi]() { fn(engine2, sqi); });
            future1.get();
            future2.get();

            engine1->ShuffleBuffers(engine2);
        }
    }
}

// This method underlies all controlled gates that can't be entirely carried out at the "meta-" level.
template <typename Qubit1Fn, typename Qubit2Fn>
void QPager::ControlledGate(bool anti, bitLenInt controlBit, bitLenInt target, Qubit2Fn cfn, Qubit1Fn fn)
{

    if (qPageCount == ONE_BCI) {
        cfn(qPages[0], controlBit, target);
        return;
    }

    if ((controlBit >= qubitsPerPage) && (target >= qubitsPerPage)) {
        // If both the control and target are "meta-," we can do this at a "meta-" level with a single-bit gate
        // "payload."
        MetaControlled(anti, { static_cast<bitLenInt>(controlBit - qubitsPerPage) },
            static_cast<bitLenInt>(target - qubitsPerPage), fn);
    } else if (controlBit >= qubitsPerPage) {
        // If the control is "meta-," but the target is not, we do this semi- at a "meta-" level, again with a
        // single-bit gate payload.
        SemiMetaControlled(anti, { static_cast<bitLenInt>(controlBit - qubitsPerPage) }, target, fn);
    } else if (controlBit == (qubitsPerPage - 1U)) {
        // There's a particular edge case not handled by any of the above, where both control and target fit in
        // independent sub-engines, but the control bit is the highest bit in the sub-engine.
        ControlledSkip(anti, 0, target, fn);
    }
}

template <typename Qubit1Fn, typename Qubit2Fn, typename Qubit3Fn>
void QPager::DoublyControlledGate(
    bool anti, bitLenInt controlBit1, bitLenInt controlBit2, bitLenInt target, Qubit3Fn ccfn, Qubit2Fn cfn, Qubit1Fn fn)
{
    if (qPageCount == ONE_BCI) {
        ccfn(qPages[0], controlBit1, controlBit2, target);
        return;
    }

    bitLenInt lowControl, highControl;
    if (controlBit1 < controlBit2) {
        lowControl = controlBit1;
        highControl = controlBit2;
    } else {
        lowControl = controlBit2;
        highControl = controlBit1;
    }

    // Like singly-controlled gates, we can handle this entirely "meta-," "semi-meta-," or parallelized entirely
    // independently within sub-engines.
    if ((lowControl >= qubitsPerPage) && (target >= qubitsPerPage)) {
        MetaControlled(anti,
            { static_cast<bitLenInt>(controlBit1 - qubitsPerPage),
                static_cast<bitLenInt>(controlBit2 - qubitsPerPage) },
            static_cast<bitLenInt>(target - qubitsPerPage), fn);
    } else if (lowControl >= qubitsPerPage) {
        // Both controls >= qubitsPerPage, target < qubitsPerPage
        SemiMetaControlled(anti,
            { static_cast<bitLenInt>(lowControl - qubitsPerPage), static_cast<bitLenInt>(highControl - qubitsPerPage) },
            target, fn);
    } else if (lowControl == (qubitsPerPage - 1U)) {
        // Again, there's a particular edge case not handled by any of the above, where the low control bit and the
        // target bit fit in independent sub-engines, but the low control bit is the highest bit in the sub-engine.
        ControlledSkip(anti, 1, target, fn);
    }
}

// This is like the QEngineCPU and QEngineOCL logic for register-like CNOT and CCNOT, just swapping sub-engine indices
// instead of amplitude indices.
template <typename Qubit1Fn>
void QPager::MetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn)
{
    bitLenInt i;

    std::vector<bitLenInt> sortedMasks(1 + controls.size());
    sortedMasks[controls.size()] = 1 << target;

    bitCapInt controlMask = 0;
    for (i = 0; i < controls.size(); i++) {
        sortedMasks[i] = 1 << controls[i];
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }

    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapInt targetMask = 1 << target;
    bitLenInt sqi = qubitsPerPage - 1;

    bitLenInt maxLCV = qPageCount >> (sortedMasks.size());
    std::vector<std::future<void>> futures(maxLCV);

    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, &sortedMasks, &controlMask, &targetMask, &sqi, fn]() {
            bitCapInt j, k, jLo, jHi;
            jHi = i;
            j = 0;
            for (k = 0; k < (sortedMasks.size()); k++) {
                jLo = jHi & sortedMasks[k];
                jHi = (jHi ^ jLo) << 1;
                j |= jLo;
            }
            j |= jHi | controlMask;

            QEnginePtr engine1 = qPages[j];
            QEnginePtr engine2 = qPages[j + targetMask];

            engine1->ShuffleBuffers(engine2);

            std::future<void> future1 = std::async(std::launch::async, [engine1, fn, sqi]() { fn(engine1, sqi); });
            std::future<void> future2 = std::async(std::launch::async, [engine2, fn, sqi]() { fn(engine2, sqi); });
            future1.get();
            future2.get();

            engine1->ShuffleBuffers(engine2);
        });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

// This is called when control bits are "meta-" but the target bit is below the "meta-" threshold, (low enough to fit in
// sub-engines).
template <typename Qubit1Fn>
void QPager::SemiMetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn)
{
    bitLenInt i;
    bitLenInt maxLCV = qPageCount >> (controls.size());
    std::vector<bitLenInt> sortedMasks(controls.size());
    bitCapInt controlMask = 0;
    for (i = 0; i < controls.size(); i++) {
        sortedMasks[i] = 1 << controls[i];
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    std::vector<std::future<void>> futures(maxLCV);
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, fn, sortedMasks, controlMask, target]() {
            bitCapInt j, k, jLo, jHi;
            jHi = i;
            j = 0;
            for (k = 0; k < (sortedMasks.size()); k++) {
                jLo = jHi & sortedMasks[k];
                jHi = (jHi ^ jLo) << 1;
                j |= jLo;
            }
            j |= jHi | controlMask;

            fn(qPages[j], target);
        });
    }
    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

// This is the particular edge case between any "meta-" gate and any gate entirely below the "meta-"/"embarrassingly
// parallel" threshold, where a control bit is the most significant bit in a sub-engine.
template <typename Qubit1Fn>
void QPager::ControlledSkip(bool anti, bitLenInt controlDepth, bitLenInt target, Qubit1Fn fn)
{
    bitLenInt i, j;
    bitLenInt k = 0;
    bitLenInt groupCount = 1 << (qubitCount - (target + 1));
    bitLenInt groupSize = 1 << ((target + 1) - qubitsPerPage);
    std::vector<std::future<void>> futures((groupCount * groupSize) / 2);
    bitLenInt sqi = qubitsPerPage - 1;
    bitLenInt jStart = (anti | (controlDepth == 0)) ? 0 : ((groupSize / 2) - 1);
    bitLenInt jInc = (controlDepth == 0) ? 1 : 2;

    for (i = 0; i < groupCount; i++) {
        for (j = jStart; j < (groupSize / 2); j += jInc) {
            futures[k] = std::async(std::launch::async, [this, groupSize, i, j, fn, sqi, anti]() {
                QEnginePtr engine1 = qPages[j + (i * groupSize)];
                QEnginePtr engine2 = qPages[j + (i * groupSize) + (groupSize / 2)];

                engine1->ShuffleBuffers(engine2);

                if (anti) {
                    fn(engine1, sqi);
                } else {
                    fn(engine2, sqi);
                }

                engine1->ShuffleBuffers(engine2);
            });
            k++;
        }
    }
    for (i = 0; i < k; i++) {
        futures[i].get();
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
        CombineEngines(highestBit);
    }

    std::vector<std::future<void>> futures(qPages.size());
    for (i = 0; i < qPages.size(); i++) {
        futures[i] = std::async(std::launch::async, [this, fn, i]() { fn(qPages[i]); });
    }
    for (i = 0; i < qPages.size(); i++) {
        futures[i].get();
    }

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

    SingleBitGate(target, [mtrx](QEnginePtr engine, bitLenInt lTarget) { engine->ApplySingleBit(mtrx, lTarget); });
}

void QPager::ApplyEitherControlledSingleBit(const bool& anti, const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex* mtrx)
{
    std::vector<bitLenInt> metaControls;
    std::vector<bitLenInt> intraControls;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (controls[i] < (qubitsPerPage - 1U)) {
            intraControls.push_back(controls[i]);
        } else {
            metaControls.push_back(controls[i]);
        }
    }

    auto dc = [anti, mtrx, &intraControls](
                  QEnginePtr engine, bitLenInt lControl1, bitLenInt lControl2, bitLenInt lTarget) {
        std::vector<bitLenInt> lControls = { lControl1, lControl2 };
        lControls.insert(lControls.end(), intraControls.begin(), intraControls.end());
        if (anti) {
            engine->ApplyAntiControlledSingleBit(&(lControls[0]), 2U + intraControls.size(), lTarget, mtrx);
        } else {
            engine->ApplyControlledSingleBit(&(lControls[0]), 2U + intraControls.size(), lTarget, mtrx);
        }
    };

    auto sc = [anti, mtrx, &intraControls](QEnginePtr engine, bitLenInt lControl1, bitLenInt lTarget) {
        std::vector<bitLenInt> lControls = { lControl1 };
        lControls.insert(lControls.end(), intraControls.begin(), intraControls.end());
        if (anti) {
            engine->ApplyAntiControlledSingleBit(&(lControls[0]), 1U + intraControls.size(), lTarget, mtrx);
        } else {
            engine->ApplyControlledSingleBit(&(lControls[0]), 1U + intraControls.size(), lTarget, mtrx);
        }
    };

    auto sg = [anti, mtrx, &intraControls](QEnginePtr engine, bitLenInt lTarget) {
        if (intraControls.size()) {
            if (anti) {
                engine->ApplyAntiControlledSingleBit(&(intraControls[0]), intraControls.size(), lTarget, mtrx);
            } else {
                engine->ApplyControlledSingleBit(&(intraControls[0]), intraControls.size(), lTarget, mtrx);
            }
        } else {
            engine->ApplySingleBit(mtrx, lTarget);
        }
    };

    if (metaControls.size() > 2U) {
        CombineEngines();
        if (anti) {
            qPages[0]->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
        } else {
            qPages[0]->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
        }
        SeparateEngines();
    } else if (metaControls.size() == 2U) {
        DoublyControlledGate(anti, metaControls[0], metaControls[1], target, dc, sc, sg);
    } else if (metaControls.size() == 1U) {
        ControlledGate(anti, metaControls[0], target, sc, sg);
    } else {
        SingleBitGate(target, sg);
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
    if (qPages.size() == 1U) {
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
