//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <future>

#include "qfactory.hpp"
#include "qpager.hpp"

namespace Qrack {

QPager::QPager(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool ignored, bool ignored2, bool useHostMem, int deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1 norm_thresh, std::vector<int> devList, bitLenInt qubitThreshold)
    : QInterface(qBitCount, rgp, false, useHardwareRNG, false, norm_thresh)
    , engine(eng)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , runningNorm(ONE_R1)
    , deviceIDs(devList)
    , thresholdQubitsPerPage(qubitThreshold)
{
#if !ENABLE_OPENCL
    if (eng == QINTERFACE_HYBRID) {
        eng = QINTERFACE_CPU;
    }
#endif

    if ((eng != QINTERFACE_CPU) && (eng != QINTERFACE_OPENCL) && (eng != QINTERFACE_HYBRID)) {
        throw std::invalid_argument(
            "QPager sub-engine type must be QINTERFACE_CPU, QINTERFACE_OPENCL or QINTERFACE_HYBRID.");
    }

#if ENABLE_OPENCL
    if ((thresholdQubitsPerPage == 0) && ((eng == QINTERFACE_OPENCL) || (eng == QINTERFACE_HYBRID))) {
        // Single bit gates act pairwise on amplitudes, so add at least 1 qubit to the log2 of the preferred
        // concurrency.
        thresholdQubitsPerPage =
            log2(OCLEngine::Instance()->GetDeviceContextPtr(devID)->GetPreferredConcurrency()) + 12U;
    }
#endif

    if (thresholdQubitsPerPage == 0) {
        // TODO: Tune for QEngineCPU
        thresholdQubitsPerPage = 18;
    }

    if (deviceIDs.size() == 0) {
        deviceIDs.push_back(devID);
    }

    SetQubitCount(qubitCount);

    if (baseQubitsPerPage > (sizeof(bitCapIntOcl) * bitsInByte)) {
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");
    }

    bool isPermInPage;
    bitCapIntOcl pagePerm = 0;
    for (bitCapIntOcl i = 0; i < basePageCount; i++) {
        isPermInPage = (initState >= pagePerm);
        pagePerm += basePageMaxQPower;
        isPermInPage &= (initState < pagePerm);
        if (isPermInPage) {
            qPages.push_back(MakeEngine(
                baseQubitsPerPage, initState - (pagePerm - basePageMaxQPower), deviceIDs[i % deviceIDs.size()]));
        } else {
            qPages.push_back(MakeEngine(baseQubitsPerPage, 0, deviceIDs[i % deviceIDs.size()]));
            qPages.back()->SetAmplitude(0, ZERO_CMPLX);
        }
    }
}

QEnginePtr QPager::MakeEngine(bitLenInt length, bitCapInt perm, int deviceId)
{
    return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engine, length, perm, rand_generator, phaseFactor,
        false, false, useHostRam, deviceId, useRDRAND, isSparse, amplitudeFloor));
}

void QPager::CombineEngines(bitLenInt bit)
{
    if (bit > qubitCount) {
        bit = qubitCount;
    }

    if ((qPages.size() == 1U) || (bit <= qubitsPerPage())) {
        return;
    }

    bitCapIntOcl groupCount = pow2Ocl(qubitCount - bit);
    bitCapIntOcl groupSize = (bitCapIntOcl)(qPages.size() / groupCount);
    bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    std::vector<QEnginePtr> nQPages;

    bitCapIntOcl i, j;

    for (i = 0; i < groupCount; i++) {
        nQPages.push_back(MakeEngine(bit, 0, deviceIDs[i % deviceIDs.size()]));
        for (j = 0; j < groupSize; j++) {
            nQPages.back()->SetAmplitudePage(qPages[j + (i * groupSize)], 0, j * pagePower, pagePower);
        }
    }

    qPages = nQPages;
}

void QPager::SeparateEngines(bitLenInt thresholdBits)
{
    if (thresholdBits < baseQubitsPerPage) {
        thresholdBits = baseQubitsPerPage;
    }

    if (thresholdBits >= qubitsPerPage()) {
        return;
    }

    bitCapIntOcl i, j;
    bitCapIntOcl pagesPer = pow2Ocl(qubitCount - thresholdBits) / qPages.size();
    bitCapIntOcl pageMaxQPower = pow2Ocl(thresholdBits);

    std::vector<QEnginePtr> nQPages;
    for (i = 0; i < qPages.size(); i++) {
        for (j = 0; j < pagesPer; j++) {
            nQPages.push_back(MakeEngine(thresholdBits, 0, deviceIDs[(j + (i * pagesPer)) % deviceIDs.size()]));
            nQPages.back()->SetAmplitudePage(qPages[i], j * pageMaxQPower, 0, pageMaxQPower);
        }
    }

    qPages = nQPages;
}

template <typename Qubit1Fn> void QPager::SingleBitGate(bitLenInt target, Qubit1Fn fn)
{
    SeparateEngines(target + 1U);

    bitLenInt qpp = qubitsPerPage();

    bitCapIntOcl i;

    if (doNormalize) {
        runningNorm = ZERO_R1;
        for (i = 0; i < qPages.size(); i++) {
            qPages[i]->Finish();
            runningNorm += qPages[i]->GetRunningNorm();
        }
        for (i = 0; i < qPages.size(); i++) {
            qPages[i]->QueueSetRunningNorm(runningNorm);
            qPages[i]->QueueSetDoNormalize(true);
        }
    }

    if (target < qpp) {
        std::vector<std::future<void>> futures(qPages.size());
        for (i = 0; i < qPages.size(); i++) {
            QEnginePtr engine = qPages[i];
            futures[i] = std::async(std::launch::async, [engine, fn, target]() {
                fn(engine, target);
                engine->QueueSetDoNormalize(false);
            });
        }
        for (i = 0; i < qPages.size(); i++) {
            futures[i].get();
        }

        return;
    }

    bitLenInt sqi = qpp - 1U;
    target -= qpp;
    bitCapIntOcl targetPow = pow2Ocl(target);
    bitCapIntOcl targetMask = targetPow - ONE_BCI;
    bitCapIntOcl maxLCV = qPages.size() >> ONE_BCI;
    std::vector<std::future<void>> futures(maxLCV);
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, fn, &targetPow, &targetMask, &sqi]() {
            bitCapIntOcl j = i & targetMask;
            j |= (i ^ j) << ONE_BCI;

            QEnginePtr engine1 = qPages[j];
            QEnginePtr engine2 = qPages[j + targetPow];

            engine1->ShuffleBuffers(engine2);

            std::future<void> future1, future2;
            future1 = std::async(std::launch::async, [fn, engine1, &sqi]() {
                fn(engine1, sqi);
                engine1->QueueSetDoNormalize(false);
            });
            future2 = std::async(std::launch::async, [fn, engine2, &sqi]() {
                fn(engine2, sqi);
                engine2->QueueSetDoNormalize(false);
            });
            future1.get();
            future2.get();

            engine1->ShuffleBuffers(engine2);
        });
    }
    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

// This is like the QEngineCPU and QEngineOCL logic for register-like CNOT and CCNOT, just swapping sub-engine indices
// instead of amplitude indices.
template <typename Qubit1Fn>
void QPager::MetaControlled(
    bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn, const complex* mtrx)
{
    bitLenInt qpp = qubitsPerPage();
    target -= qpp;
    bitLenInt sqi = qpp - 1U;

    std::vector<bitCapIntOcl> sortedMasks(1U + controls.size());
    bitCapIntOcl targetPow = pow2Ocl(target);
    sortedMasks[controls.size()] = targetPow - ONE_BCI;

    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controls.size(); i++) {
        sortedMasks[i] = pow2Ocl(controls[i] - qpp);
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bool isSpecial, isInvert;
    complex top, bottom;
    if ((mtrx[1] == ZERO_CMPLX) && (mtrx[2] == ZERO_CMPLX)) {
        isSpecial = true;
        isInvert = false;
        top = mtrx[0];
        bottom = mtrx[3];
    } else if ((mtrx[0] == ZERO_CMPLX) && (mtrx[3] == ZERO_CMPLX)) {
        isSpecial = true;
        isInvert = true;
        top = mtrx[1];
        bottom = mtrx[2];
    } else {
        isSpecial = false;
        isInvert = false;
        top = ZERO_CMPLX;
        bottom = ZERO_CMPLX;
    }

    bitCapIntOcl maxLCV = qPages.size() >> sortedMasks.size();
    std::vector<std::future<void>> futures(maxLCV);
    bitCapIntOcl i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async,
            [this, i, fn, &sqi, &controlMask, &targetPow, &sortedMasks, &isSpecial, &isInvert, &top, &bottom]() {
                bitCapIntOcl j, k, jLo, jHi;
                jHi = i;
                j = 0;
                for (k = 0; k < (sortedMasks.size()); k++) {
                    jLo = jHi & sortedMasks[k];
                    jHi = (jHi ^ jLo) << ONE_BCI;
                    j |= jLo;
                }
                j |= jHi | controlMask;

                if (isSpecial && isInvert) {
                    std::swap(qPages[j], qPages[j + targetPow]);
                }

                QEnginePtr engine1 = qPages[j];
                QEnginePtr engine2 = qPages[j + targetPow];

                std::future<void> future1, future2;
                if (isSpecial) {
                    if (top != ONE_CMPLX) {
                        future1 = std::async(
                            std::launch::async, [engine1, &top]() { engine1->ApplySinglePhase(top, top, 0); });
                    }
                    if (bottom != ONE_CMPLX) {
                        future2 = std::async(
                            std::launch::async, [engine2, &bottom]() { engine2->ApplySinglePhase(bottom, bottom, 0); });
                    }

                    if (top != ONE_CMPLX) {
                        future1.get();
                    }
                    if (bottom != ONE_CMPLX) {
                        future2.get();
                    }
                } else {
                    engine1->ShuffleBuffers(engine2);

                    future1 = std::async(std::launch::async, [engine1, fn, &sqi]() { fn(engine1, sqi); });
                    future2 = std::async(std::launch::async, [engine2, fn, &sqi]() { fn(engine2, sqi); });
                    future1.get();
                    future2.get();

                    engine1->ShuffleBuffers(engine2);
                }
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
    bitLenInt qpp = qubitsPerPage();

    std::vector<bitLenInt> sortedMasks(controls.size());

    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controls.size(); i++) {
        sortedMasks[i] = pow2Ocl(controls[i] - qpp);
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapIntOcl maxLCV = qPages.size() >> sortedMasks.size();
    std::vector<std::future<void>> futures(maxLCV);
    bitCapIntOcl i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, fn, &controlMask, &target, &sortedMasks]() {
            bitCapIntOcl j, k, jLo, jHi;
            jHi = i;
            j = 0;
            for (k = 0; k < (sortedMasks.size()); k++) {
                jLo = jHi & sortedMasks[k];
                jHi = (jHi ^ jLo) << ONE_BCI;
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

template <typename F> void QPager::CombineAndOp(F fn, std::vector<bitLenInt> bits)
{
    if (qPages.size() == 1U) {
        fn(qPages[0]);
        return;
    }

    bitLenInt highestBit = 0;
    for (bitLenInt i = 0; i < bits.size(); i++) {
        if (bits[i] > highestBit) {
            highestBit = bits[i];
        }
    }

    if (highestBit >= qubitsPerPage()) {
        CombineEngines(highestBit + 1U);
    } else {
        // Lazy separate: avoid cycling through combine/separate in successive CombineAndOp() calls
        SeparateEngines(highestBit + 1U);
    }

    std::vector<std::future<void>> futures(qPages.size());
    bitCapIntOcl i;
    for (i = 0; i < qPages.size(); i++) {
        futures[i] = std::async(std::launch::async, [this, fn, i]() { fn(qPages[i]); });
    }
    for (i = 0; i < qPages.size(); i++) {
        futures[i].get();
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
    toCopy->CombineEngines();
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->Compose(toCopy->qPages[0]);
    }
    bitLenInt toRet = qubitCount;
    SetQubitCount(qubitCount + toCopy->qubitCount);
    return toRet;
}

bitLenInt QPager::Compose(QPagerPtr toCopy, bitLenInt start)
{
    if (start == qubitCount) {
        return Compose(toCopy);
    }

    bitLenInt inPage = qubitCount - start;

    CombineEngines(inPage);
    toCopy->CombineEngines();
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->Compose(toCopy->qPages[0], qPages[i]->GetQubitCount() - inPage);
    }
    SetQubitCount(qubitCount + toCopy->qubitCount);

    return inPage;
}

void QPager::Decompose(bitLenInt start, QPagerPtr dest)
{
    CombineEngines(start + dest->qubitCount);
    dest->CombineEngines();
    qPages[0]->Decompose(start, dest->qPages[0]);
    // To be clear, under the assumption of perfect decomposibility, all further pages should produce the exact same
    // "dest" as the line above, hence we can take just the first one and "Dispose" the rest. (This might pose a
    // problem or limitation for "approximate separability.")
    for (bitCapIntOcl i = 1; i < qPages.size(); i++) {
        qPages[i]->Dispose(start, dest->qubitCount);
    }
    SetQubitCount(qubitCount - dest->qubitCount);
}

void QPager::Dispose(bitLenInt start, bitLenInt length)
{
    CombineEngines(start + length);
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->Dispose(start, length);
    }
    SetQubitCount(qubitCount - length);
}

void QPager::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    CombineEngines(start + length);
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->Dispose(start, length, disposedPerm);
    }
    SetQubitCount(qubitCount - length);
}

void QPager::SetQuantumState(const complex* inputState)
{
    bitCapIntOcl pagePerm = 0;
    bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->SetQuantumState(inputState + pagePerm);
        if (doNormalize) {
            qPages[i]->UpdateRunningNorm();
        }
        pagePerm += pagePower;
    }
}

void QPager::GetQuantumState(complex* outputState)
{
    bitCapIntOcl pagePerm = 0;
    bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->GetQuantumState(outputState + pagePerm);
        pagePerm += pagePower;
    }
}

void QPager::GetProbs(real1* outputProbs)
{
    bitCapIntOcl pagePerm = 0;
    bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->GetProbs(outputProbs + pagePerm);
        pagePerm += pagePower;
    }
}

void QPager::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool isPermInPage;
    bitCapIntOcl pagePerm = 0;
    bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        isPermInPage = (perm >= pagePerm);
        pagePerm += pagePower;
        isPermInPage &= (perm < pagePerm);

        if (isPermInPage) {
            qPages[i]->SetPermutation(perm - (pagePerm - pagePower));
            continue;
        }

        qPages[i]->ZeroAmplitudes();
    }
}

void QPager::ApplySingleBit(const complex* mtrx, bitLenInt target)
{
    SingleBitGate(target, [mtrx](QEnginePtr engine, bitLenInt lTarget) { engine->ApplySingleBit(mtrx, lTarget); });
}

void QPager::ApplySingleEither(const bool& isInvert, complex top, complex bottom, bitLenInt target)
{
    if (target < qubitsPerPage()) {
        SeparateEngines(target + 1U);
        if (isInvert) {
            SingleBitGate(target, [top, bottom](QEnginePtr engine, bitLenInt lTarget) {
                engine->ApplySingleInvert(top, bottom, lTarget);
            });
        } else {
            SingleBitGate(target, [top, bottom](QEnginePtr engine, bitLenInt lTarget) {
                engine->ApplySinglePhase(top, bottom, lTarget);
            });
        }
        return;
    }

    SeparateEngines();
    bitLenInt qpp = qubitsPerPage();

    target -= qpp;
    bitCapIntOcl targetPow = pow2Ocl(target);
    bitCapIntOcl qMask = targetPow - 1U;

    if (randGlobalPhase) {
        bottom /= top;
        top = ONE_CMPLX;
    }

    bitCapIntOcl maxLCV = qPages.size() >> 1U;
    std::vector<std::future<void>> futures(maxLCV);
    bitCapIntOcl i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, &isInvert, &top, &bottom, &targetPow, &qMask]() {
            bitCapIntOcl j = i & qMask;
            j |= (i ^ j) << ONE_BCI;

            if (isInvert) {
                std::swap(qPages[j], qPages[j + targetPow]);
            }

            QEnginePtr engine1 = qPages[j];
            QEnginePtr engine2 = qPages[j + targetPow];

            std::future<void> future1, future2;
            if (top != ONE_CMPLX) {
                future1 = std::async(std::launch::async, [engine1, top]() { engine1->ApplySinglePhase(top, top, 0); });
            }
            if (bottom != ONE_CMPLX) {
                future2 = std::async(
                    std::launch::async, [engine2, bottom]() { engine2->ApplySinglePhase(bottom, bottom, 0); });
            }

            if (top != ONE_CMPLX) {
                future1.get();
            }
            if (bottom != ONE_CMPLX) {
                future2.get();
            }
        });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

void QPager::ApplyEitherControlledSingleBit(const bool& anti, const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex* mtrx)
{
    if (controlLen == 0) {
        ApplySingleBit(mtrx, target);
        return;
    }

    SeparateEngines(target + 1U);

    bitLenInt qpp = qubitsPerPage();

    std::vector<bitLenInt> metaControls;
    std::vector<bitLenInt> intraControls;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if (controls[i] < qpp) {
            intraControls.push_back(controls[i]);
        } else {
            metaControls.push_back(controls[i]);
        }
    }

    auto sg = [anti, mtrx, intraControls](QEnginePtr engine, bitLenInt lTarget) {
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

    if (metaControls.size() == 0) {
        SingleBitGate(target, sg);
    } else if (target < qpp) {
        SemiMetaControlled(anti, metaControls, target, sg);
    } else {
        MetaControlled(anti, metaControls, target, sg, mtrx);
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

void QPager::UniformParityRZ(const bitCapInt& mask, const real1& angle)
{
    // TODO: Identify highest bit, and CombineAndOp()
    CombineEngines();
    qPages[0]->UniformParityRZ(mask, angle);
}

void QPager::CUniformParityRZ(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1& angle)
{
    // TODO: Identify highest bit, and CombineAndOp()
    CombineEngines();
    qPages[0]->CUniformParityRZ(controls, controlLen, mask, angle);
}

void QPager::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    if (controlLen == 0) {
        Swap(qubit1, qubit2);
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->CSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (controlLen == 0) {
        Swap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (controlLen == 0) {
        SqrtSwap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->CSqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (controlLen == 0) {
        SqrtSwap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (controlLen == 0) {
        ISqrtSwap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->CISqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (controlLen == 0) {
        ISqrtSwap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}

bool QPager::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    CombineEngines();
    bool toRet = qPages[0]->ForceM(qubit, result, doForce, doApply);
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
    SeparateEngines(start + length);

    bitLenInt qpp = qubitsPerPage();

    bitCapIntOcl i;

    if (start >= qpp) {
        // Entirely meta-
        start -= qpp;
        bitCapIntOcl mask = pow2Ocl(start + length) - pow2Ocl(start);
        std::vector<std::future<void>> futures;
        for (i = 0; i < qPages.size(); i++) {
            if ((i & mask) == 0U) {
                QInterfacePtr engine = qPages[i];
                futures.push_back(std::async(std::launch::async, [engine]() { engine->PhaseFlip(); }));
            }
        }

        for (i = 0; i < futures.size(); i++) {
            futures[i].get();
        }

        return;
    }

    if ((start + length) >= qpp) {
        // Semi-meta-
        bitLenInt metaLen = (start + length) - qpp;
        bitLenInt remainderLen = length - metaLen;
        bitCapIntOcl mask = pow2Ocl(metaLen) - ONE_BCI;
        std::vector<std::future<void>> futures;
        for (i = 0; i < qPages.size(); i++) {
            if ((i & mask) == 0U) {
                QInterfacePtr engine = qPages[i];
                futures.push_back(std::async(std::launch::async,
                    [engine, &start, &remainderLen]() { engine->ZeroPhaseFlip(start, remainderLen); }));
            }
        }

        for (i = 0; i < futures.size(); i++) {
            futures[i].get();
        }

        return;
    }

    // Contained in sub-units
    std::vector<std::future<void>> futures(qPages.size());
    for (i = 0; i < qPages.size(); i++) {
        QInterfacePtr engine = qPages[i];
        futures[i] =
            std::async(std::launch::async, [engine, &start, &length]() { engine->ZeroPhaseFlip(start, length); });
    }
    for (i = 0; i < qPages.size(); i++) {
        futures[i].get();
    }
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
    for (bitLenInt i = 0; i < qPages.size(); i++) {
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

void QPager::MetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac)
{
    bitLenInt qpp = qubitsPerPage();
    qubit1 -= qpp;
    qubit2 -= qpp;

    std::vector<bitCapIntOcl> sortedMasks(2U);
    bitCapIntOcl qubit1Pow = pow2Ocl(qubit1);
    sortedMasks[0] = qubit1Pow - ONE_BCI;
    bitCapIntOcl qubit2Pow = pow2Ocl(qubit2);
    sortedMasks[1] = qubit2Pow - ONE_BCI;
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapIntOcl maxLCV = qPages.size() >> sortedMasks.size();
    std::vector<std::future<void>> futures(maxLCV);
    bitCapIntOcl i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, &qubit1Pow, &qubit2Pow, &sortedMasks, &isIPhaseFac]() {
            bitCapIntOcl j, jLo, jHi;
            j = i & sortedMasks[0];
            jHi = (i ^ j) << ONE_BCI;
            jLo = jHi & sortedMasks[1];
            j |= jLo | ((jHi ^ jLo) << ONE_BCI);

            std::swap(qPages[j + qubit1Pow], qPages[j + qubit2Pow]);

            if (!isIPhaseFac) {
                return;
            }

            QEnginePtr engine1 = qPages[j + qubit1Pow];
            QEnginePtr engine2 = qPages[j + qubit2Pow];

            std::future<void> future1, future2;

            future1 = std::async(std::launch::async, [engine1]() { engine1->ApplySinglePhase(I_CMPLX, I_CMPLX, 0); });
            future2 = std::async(std::launch::async, [engine2]() { engine2->ApplySinglePhase(I_CMPLX, I_CMPLX, 0); });

            future1.get();
            future2.get();
        });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

void QPager::SemiMetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac)
{
    if (qubit1 > qubit2) {
        std::swap(qubit1, qubit2);
    }

    bitLenInt qpp = qubitsPerPage();
    qubit2 -= qpp;
    bitLenInt sqi = qpp - 1U;

    bitCapIntOcl qubit2Pow = pow2Ocl(qubit2);
    bitCapIntOcl qubit2Mask = qubit2Pow - ONE_BCI;

    bitCapIntOcl maxLCV = qPages.size() >> ONE_BCI;
    std::vector<std::future<void>> futures(maxLCV);
    bitCapIntOcl i;
    for (i = 0; i < maxLCV; i++) {
        futures[i] = std::async(std::launch::async, [this, i, &qubit1, &qubit2Pow, &qubit2Mask, &isIPhaseFac, &sqi]() {
            bitCapIntOcl j = i & qubit2Mask;
            j |= (i ^ j) << ONE_BCI;

            QEnginePtr engine1 = qPages[j];
            QEnginePtr engine2 = qPages[j + qubit2Pow];

            engine1->ShuffleBuffers(engine2);

            std::future<void> future1, future2;

            if (qubit1 == sqi) {
                if (isIPhaseFac) {
                    future1 = std::async(
                        std::launch::async, [engine1, &sqi]() { engine1->ApplySinglePhase(ZERO_CMPLX, I_CMPLX, sqi); });
                    future2 = std::async(
                        std::launch::async, [engine2, &sqi]() { engine2->ApplySinglePhase(I_CMPLX, ZERO_CMPLX, sqi); });

                    future1.get();
                    future2.get();
                }
                return;
            }

            if (isIPhaseFac) {
                future1 = std::async(std::launch::async, [engine1, &qubit1, &sqi]() { engine1->ISwap(qubit1, sqi); });
                future2 = std::async(std::launch::async, [engine2, &qubit1, &sqi]() { engine2->ISwap(qubit1, sqi); });
            } else {
                future1 = std::async(std::launch::async, [engine1, &qubit1, &sqi]() { engine1->Swap(qubit1, sqi); });
                future2 = std::async(std::launch::async, [engine2, &qubit1, &sqi]() { engine2->Swap(qubit1, sqi); });
            }

            future1.get();
            future2.get();

            engine1->ShuffleBuffers(engine2);
        });
    }

    for (i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
}

void QPager::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    bitLenInt qpp = baseQubitsPerPage;
    bool isQubit1Meta = qubit1 >= qpp;
    bool isQubit2Meta = qubit2 >= qpp;
    if (isQubit1Meta && isQubit2Meta) {
        SeparateEngines();
        MetaSwap(qubit1, qubit2, false);
        return;
    }
    if (isQubit1Meta || isQubit2Meta) {
        SeparateEngines();
        SemiMetaSwap(qubit1, qubit2, false);
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->Swap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::ISwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    bitLenInt qpp = baseQubitsPerPage;
    bool isQubit1Meta = qubit1 >= qpp;
    bool isQubit2Meta = qubit2 >= qpp;
    if (isQubit1Meta && isQubit2Meta) {
        SeparateEngines();
        MetaSwap(qubit1, qubit2, true);
        return;
    }
    if (isQubit1Meta || isQubit2Meta) {
        SeparateEngines();
        SemiMetaSwap(qubit1, qubit2, true);
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->ISwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->SqrtSwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->ISqrtSwap(qubit1, qubit2); }, { qubit1, qubit2 });
}
void QPager::FSim(real1 theta, real1 phi, bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->FSim(theta, phi, qubit1, qubit2); }, { qubit1, qubit2 });
}

real1 QPager::Prob(bitLenInt qubitIndex)
{
    if (qubitIndex >= qubitsPerPage()) {
        CombineEngines(qubitIndex + 1U);
    } else {
        SeparateEngines(qubitIndex + 1U);
    }

    if (qPages.size() == 1U) {
        return qPages[0]->Prob(qubitIndex);
    }

    real1 oneChance = ZERO_R1;
    bitCapIntOcl i;

    std::vector<std::future<real1>> futures(qPages.size());
    for (i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
        futures[i] = std::async(std::launch::async, [engine, qubitIndex]() { return engine->Prob(qubitIndex); });
    }
    for (i = 0; i < qPages.size(); i++) {
        oneChance += futures[i].get();
    }

    return oneChance;
}
real1 QPager::ProbAll(bitCapInt fullRegister)
{
    bitCapIntOcl subIndex = (bitCapIntOcl)(fullRegister / pageMaxQPower());
    fullRegister -= subIndex * pageMaxQPower();
    return qPages[subIndex]->ProbAll(fullRegister);
}
real1 QPager::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    CombineEngines();
    real1 maskChance = qPages[0]->ProbMask(mask, permutation);
    return maskChance;
}

bool QPager::ApproxCompare(QInterfacePtr toCompare)
{
    QPagerPtr toComparePager = std::dynamic_pointer_cast<QPager>(toCompare);
    CombineEngines();
    toComparePager->CombineEngines();
    bool toRet = qPages[0]->ApproxCompare(toComparePager->qPages[0]);
    return toRet;
}
void QPager::UpdateRunningNorm(real1 norm_thresh)
{
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->UpdateRunningNorm(norm_thresh);
    }
}

QInterfacePtr QPager::Clone()
{
    bitLenInt qpp = qubitsPerPage();

    QPagerPtr clone = std::dynamic_pointer_cast<QPager>(
        CreateQuantumInterface(QINTERFACE_QPAGER, engine, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
            randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse));

    clone->CombineEngines(qpp);

    bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        clone->qPages[i]->SetAmplitudePage(qPages[i], 0, 0, pagePower);
    }

    return clone;
}

} // namespace Qrack
