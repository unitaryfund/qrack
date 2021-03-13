//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#include <future>
#include <string>

#include "qfactory.hpp"
#include "qpager.hpp"

namespace Qrack {

QPager::QPager(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool ignored, bool ignored2, bool useHostMem, int deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1_f norm_thresh, std::vector<int> devList, bitLenInt qubitThreshold)
    : QInterface(qBitCount, rgp, false, useHardwareRNG, false, norm_thresh)
    , engine(eng)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , runningNorm(ONE_R1)
    , deviceIDs(devList)
    , useHardwareThreshold(false)
    , thresholdQubitsPerPage(qubitThreshold)
{
#if !ENABLE_OPENCL
    if (engine == QINTERFACE_HYBRID) {
        eng = QINTERFACE_CPU;
    }
#endif

    if ((engine != QINTERFACE_CPU) && (engine != QINTERFACE_OPENCL) && (engine != QINTERFACE_HYBRID)) {
        throw std::invalid_argument(
            "QPager sub-engine type must be QINTERFACE_CPU, QINTERFACE_OPENCL or QINTERFACE_HYBRID.");
    }

    bitLenInt qpd = 2U;
    if (getenv("QRACK_DEVICE_GLOBAL_QB")) {
        qpd = (bitLenInt)std::stoi(std::string(getenv("QRACK_DEVICE_GLOBAL_QB")));
    }

#if ENABLE_OPENCL
    if ((thresholdQubitsPerPage == 0) && ((engine == QINTERFACE_OPENCL) || (engine == QINTERFACE_HYBRID))) {
        useHardwareThreshold = true;

        // Limit at the power of 2 less-than-or-equal-to a full max memory allocation segment, or choose with
        // environment variable.

        bitLenInt pps = 0;
        if (getenv("QRACK_SEGMENT_GLOBAL_QB")) {
            pps = (bitLenInt)std::stoi(std::string(getenv("QRACK_SEGMENT_GLOBAL_QB")));
        }

        maxPageQubits = log2(OCLEngine::Instance()->GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex)) - pps;

        thresholdQubitsPerPage = maxPageQubits;

        if ((qubitCount - qpd) < thresholdQubitsPerPage) {
            thresholdQubitsPerPage = qubitCount - qpd;
        }

        // Single bit gates act pairwise on amplitudes, so add at least 1 qubit to the log2 of the preferred
        // concurrency.
        minPageQubits = log2(OCLEngine::Instance()->GetDeviceContextPtr(devID)->GetPreferredConcurrency()) + 2U;

        if (thresholdQubitsPerPage < minPageQubits) {
            thresholdQubitsPerPage = minPageQubits;
        }
    }
#endif

    if (thresholdQubitsPerPage == 0) {
        useHardwareThreshold = true;

        thresholdQubitsPerPage = qubitCount - qpd;

        maxPageQubits = -1;
        minPageQubits = log2(std::thread::hardware_concurrency()) + PSTRIDEPOW;

        if (thresholdQubitsPerPage < minPageQubits) {
            thresholdQubitsPerPage = minPageQubits;
        }
    }

    if (deviceIDs.size() == 0) {
        deviceIDs.push_back(devID);
    }

    SetQubitCount(qubitCount);

    if (baseQubitsPerPage > (sizeof(bitCapIntOcl) * bitsInByte)) {
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");
    }

    initState &= maxQPower - ONE_BCI;
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
        false, false, useHostRam, deviceId, useRDRAND, isSparse, (real1_f)amplitudeFloor));
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
        QEnginePtr engine = nQPages.back();
        for (j = 0; j < groupSize; j++) {
            engine->SetAmplitudePage(qPages[j + (i * groupSize)], 0, j * pagePower, pagePower);
        }
    }

    qPages = nQPages;
}

void QPager::SeparateEngines(bitLenInt thresholdBits, bool noBaseFloor)
{
    if (!noBaseFloor && (thresholdBits < baseQubitsPerPage)) {
        thresholdBits = baseQubitsPerPage;
    }

    if (thresholdBits >= qubitsPerPage()) {
        return;
    }

    bitCapIntOcl pagesPer = pow2Ocl(qubitCount - thresholdBits) / qPages.size();
    bitCapIntOcl pageMaxQPower = pow2Ocl(thresholdBits);
    bitCapIntOcl i, j;

    std::vector<QEnginePtr> nQPages;
    for (i = 0; i < qPages.size(); i++) {
        for (j = 0; j < pagesPer; j++) {
            nQPages.push_back(MakeEngine(thresholdBits, 0, deviceIDs[(j + (i * pagesPer)) % deviceIDs.size()]));
            nQPages.back()->SetAmplitudePage(qPages[i], j * pageMaxQPower, 0, pageMaxQPower);
        }
    }

    qPages = nQPages;
}

template <typename Qubit1Fn>
void QPager::SingleBitGate(bitLenInt target, Qubit1Fn fn, const bool& isSqiCtrl, const bool& isAnti)
{
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
        for (i = 0; i < qPages.size(); i++) {
            QEnginePtr engine = qPages[i];
            fn(engine, target);
            engine->QueueSetDoNormalize(false);
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
        futures[i] =
            std::async(std::launch::async, [this, i, fn, &targetPow, &targetMask, &sqi, &isSqiCtrl, &isAnti]() {
                bitCapIntOcl j = i & targetMask;
                j |= (i ^ j) << ONE_BCI;

                QEnginePtr engine1 = qPages[j];
                QEnginePtr engine2 = qPages[j + targetPow];

                engine1->ShuffleBuffers(engine2);

                if (!isSqiCtrl || isAnti) {
                    fn(engine1, sqi);
                }
                engine1->QueueSetDoNormalize(false);

                if (!isSqiCtrl || !isAnti) {
                    fn(engine2, sqi);
                }
                engine2->QueueSetDoNormalize(false);

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
void QPager::MetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn,
    const complex* mtrx, const bool& isSqiCtrl)
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
            [this, i, fn, &sqi, &controlMask, &targetPow, &sortedMasks, &isSpecial, &isInvert, &top, &bottom,
                &isSqiCtrl, &anti]() {
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

                if (isSpecial) {
                    bool doTop = (top != ONE_CMPLX) && (!isSqiCtrl || anti);
                    bool doBottom = (bottom != ONE_CMPLX) && (!isSqiCtrl || !anti);

                    if (doTop) {
                        engine1->ApplySinglePhase(top, top, 0);
                    }
                    if (doBottom) {
                        engine2->ApplySinglePhase(bottom, bottom, 0);
                    }
                } else {
                    engine1->ShuffleBuffers(engine2);

                    bool doTop = !isSqiCtrl || anti;
                    bool doBottom = !isSqiCtrl || !anti;

                    if (doTop) {
                        fn(engine1, sqi);
                    }
                    if (doBottom) {
                        fn(engine2, sqi);
                    }

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
    bitCapIntOcl i;
    for (i = 0; i < maxLCV; i++) {
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

    bitCapIntOcl i;
    for (i = 0; i < qPages.size(); i++) {
        fn(qPages[i]);
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
    bitLenInt qpp = qubitsPerPage();
    bitLenInt tcqpp = toCopy->qubitsPerPage();

    if ((qpp + tcqpp) > maxPageQubits) {
        tcqpp = (tcqpp < (maxPageQubits - qpp)) ? maxPageQubits - qpp : 1U;
        toCopy->SeparateEngines(tcqpp, true);
    }

    if ((qpp + tcqpp) > maxPageQubits) {
        SeparateEngines((tcqpp < qpp) ? (qpp - tcqpp) : 1U, true);
    }

    bitCapIntOcl i, j;
    bitCapInt maxJ = (toCopy->qPages.size() - 1U);
    std::vector<QEnginePtr> nQPages;

    for (i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
        for (j = 0; j < maxJ; j++) {
            nQPages.push_back(std::dynamic_pointer_cast<QEngine>(engine->Clone()));
            nQPages.back()->Compose(toCopy->qPages[j]);
        }
        nQPages.push_back(engine);
        nQPages.back()->Compose(toCopy->qPages[maxJ]);
    }

    qPages = nQPages;

    bitLenInt toRet = qubitCount;
    SetQubitCount(qubitCount + toCopy->qubitCount);

    return toRet;
}

bitLenInt QPager::Compose(QPagerPtr toCopy, bitLenInt start)
{
    if (start == qubitCount) {
        return Compose(toCopy);
    }

    toCopy->CombineEngines();

    bitLenInt qpp = qubitsPerPage();
    if ((qpp + toCopy->qubitCount) > maxPageQubits) {
        SeparateEngines((toCopy->qubitCount < qpp) ? (qpp - toCopy->qubitCount) : 1U, true);
    }

    bitLenInt inPage = qubitCount - start;

    if (start <= inPage) {
        CombineEngines(start);
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Compose(toCopy->qPages[0], start);
        }
    } else {
        CombineEngines(inPage);
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Compose(toCopy->qPages[0], qPages[i]->GetQubitCount() - inPage);
        }
    }
    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}

void QPager::Decompose(bitLenInt start, QPagerPtr dest)
{
    dest->CombineEngines();

    bitLenInt inPage = qubitCount - (start + dest->qubitCount);
    bool didDecompose = false;

    if (start <= inPage) {
        if (start == 0) {
            CombineEngines(start + dest->qubitCount + 1U);
        } else {
            CombineEngines(start + dest->qubitCount);
        }
        // To be clear, under the assumption of perfect decomposibility, all further pages should produce the exact same
        // "dest" as the line above, hence we can take just the first nonzero one and "Dispose" the rest. (This might
        // pose a problem or limitation for "approximate separability.")
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            if (!didDecompose && !qPages[i]->IsZeroAmplitude()) {
                qPages[i]->Decompose(start, dest->qPages[0]);
                didDecompose = true;
            } else {
                qPages[i]->Dispose(start, dest->qubitCount);
            }
        }
    } else {
        if ((qPages[0]->GetQubitCount() - (inPage + dest->qubitCount)) == 0) {
            CombineEngines(inPage + dest->qubitCount + 1U);
        } else {
            CombineEngines(inPage + dest->qubitCount);
        }
        // (Same as above)
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Dispose(qPages[i]->GetQubitCount() - (inPage + dest->qubitCount), dest->qubitCount);
            if (!didDecompose && !qPages[i]->IsZeroAmplitude()) {
                qPages[i]->Decompose(qPages[i]->GetQubitCount() - (inPage + dest->qubitCount), dest->qPages[0]);
                didDecompose = true;
            } else {
                qPages[i]->Dispose(qPages[i]->GetQubitCount() - (inPage + dest->qubitCount), dest->qubitCount);
            }
        }
    }

    SetQubitCount(qubitCount - dest->qubitCount);

    CombineEngines(baseQubitsPerPage);
}

void QPager::Dispose(bitLenInt start, bitLenInt length)
{
    bitLenInt inPage = qubitCount - (start + length);

    if (start <= inPage) {
        if (start == 0) {
            CombineEngines(start + length + 1U);
        } else {
            CombineEngines(start + length);
        }
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Dispose(start, length);
        }
    } else {
        if ((qPages[0]->GetQubitCount() - (inPage + length)) == 0) {
            CombineEngines(inPage + length + 1U);
        } else {
            CombineEngines(inPage + length);
        }
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Dispose(qPages[i]->GetQubitCount() - (inPage + length), length);
        }
    }

    SetQubitCount(qubitCount - length);

    CombineEngines(baseQubitsPerPage);
}

void QPager::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    bitLenInt inPage = qubitCount - (start + length);

    if (start <= inPage) {
        if (start == 0) {
            CombineEngines(start + length + 1U);
        } else {
            CombineEngines(start + length);
        }
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Dispose(start, length, disposedPerm);
        }
    } else {
        if ((qPages[0]->GetQubitCount() - (inPage + length)) == 0) {
            CombineEngines(inPage + length + 1U);
        } else {
            CombineEngines(inPage + length);
        }
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Dispose(qPages[i]->GetQubitCount() - (inPage + length), length, disposedPerm);
        }
    }

    SetQubitCount(qubitCount - length);

    CombineEngines(baseQubitsPerPage);
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
    perm &= maxQPower - ONE_BCI;
    bool isPermInPage;
    bitCapIntOcl pagePerm = 0;
    bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        isPermInPage = (perm >= pagePerm);
        pagePerm += pagePower;
        isPermInPage &= (perm < pagePerm);

        if (isPermInPage) {
            qPages[i]->SetPermutation(perm - (pagePerm - pagePower), phaseFac);
            continue;
        }

        qPages[i]->ZeroAmplitudes();
    }
}

void QPager::ApplySingleBit(const complex* mtrx, bitLenInt target)
{
    SeparateEngines();
    SingleBitGate(target, [mtrx](QEnginePtr engine, bitLenInt lTarget) { engine->ApplySingleBit(mtrx, lTarget); });
}

void QPager::ApplySingleEither(const bool& isInvert, complex top, complex bottom, bitLenInt target)
{
    SeparateEngines();
    bitLenInt qpp = qubitsPerPage();

    if (target < qpp) {
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

    if (randGlobalPhase) {
        bottom /= top;
        top = ONE_CMPLX;
    }

    target -= qpp;
    bitCapIntOcl targetPow = pow2Ocl(target);
    bitCapIntOcl qMask = targetPow - 1U;

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

            if (top != ONE_CMPLX) {
                qPages[j]->ApplySinglePhase(top, top, 0);
            }
            if (bottom != ONE_CMPLX) {
                qPages[j + targetPow]->ApplySinglePhase(bottom, bottom, 0);
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
    bool isSqiCtrl = false;
    for (bitLenInt i = 0; i < controlLen; i++) {
        if ((target >= qpp) && (controls[i] == (qpp - 1U))) {
            isSqiCtrl = true;
        } else if (controls[i] < qpp) {
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
        SingleBitGate(target, sg, isSqiCtrl, anti);
    } else if (target < qpp) {
        SemiMetaControlled(anti, metaControls, target, sg);
    } else {
        MetaControlled(anti, metaControls, target, sg, mtrx, isSqiCtrl);
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

void QPager::UniformParityRZ(const bitCapInt& mask, const real1_f& angle)
{
    CombineAndOp([&](QEnginePtr engine) { engine->UniformParityRZ(mask, angle); }, { log2(mask) });
}

void QPager::CUniformParityRZ(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CUniformParityRZ(controls, controlLen, mask, angle); },
        { log2(mask) }, controls, controlLen);
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
    real1 oneChance = Prob(qubit);
    if (!doForce) {
        if (oneChance >= ONE_R1) {
            result = true;
        } else if (oneChance <= ZERO_R1) {
            result = false;
        } else {
            real1 prob = Rand();
            result = (prob <= oneChance);
        }
    }

    real1 nrmlzr;
    if (result) {
        nrmlzr = oneChance;
    } else {
        nrmlzr = ONE_R1 - oneChance;
    }

    if (nrmlzr <= ZERO_R1) {
        throw "ERROR: Forced a measurement result with 0 probability";
    }

    if (doApply && (nrmlzr != ONE_BCI)) {
        bitLenInt qpp = qubitsPerPage();
        std::vector<std::future<void>> futures(qPages.size());
        bitCapIntOcl i;
        if (qubit < qpp) {
            complex nrmFac = GetNonunitaryPhase() / (real1)std::sqrt(nrmlzr);
            bitCapIntOcl qPower = pow2Ocl(qubit);
            for (i = 0; i < qPages.size(); i++) {
                QEnginePtr engine = qPages[i];
                futures[i] = (std::async(std::launch::async,
                    [engine, qPower, result, nrmFac]() { engine->ApplyM(qPower, result, nrmFac); }));
            }
        } else {
            bitLenInt metaQubit = qubit - qpp;
            bitCapIntOcl qPower = pow2Ocl(metaQubit);
            for (i = 0; i < qPages.size(); i++) {
                QEnginePtr engine = qPages[i];
                if (!(i & qPower) == !result) {
                    futures[i] =
                        (std::async(std::launch::async, [engine, nrmlzr]() { engine->NormalizeState(nrmlzr); }));
                } else {
                    futures[i] = (std::async(std::launch::async, [engine]() { engine->ZeroAmplitudes(); }));
                }
            }
        }

        for (i = 0; i < qPages.size(); i++) {
            futures[i].get();
        }
    }

    return result;
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
#if ENABLE_BCD
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
void QPager::DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->DECBCDC(toSub, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
#endif
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
    if (!controlLen) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) },
        controls, controlLen);
}
void QPager::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
    bitLenInt controlLen)
{
    if (!controlLen) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inOutStart + length - 1U), static_cast<bitLenInt>(carryStart + length - 1U) },
        controls, controlLen);
}
void QPager::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        MULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
}
void QPager::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        IMULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
}
void QPager::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        POWModNOut(base, modN, inStart, outStart, length);
        return;
    }

    CombineAndOpControlled(
        [&](QEnginePtr engine) { engine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen); },
        { static_cast<bitLenInt>(inStart + length - 1U), static_cast<bitLenInt>(outStart + length - 1U) }, controls,
        controlLen);
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

bitCapInt QPager::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    unsigned char* values, bool ignored)
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

            qPages[j + qubit1Pow]->ApplySinglePhase(I_CMPLX, I_CMPLX, 0);
            qPages[j + qubit2Pow]->ApplySinglePhase(I_CMPLX, I_CMPLX, 0);
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

            if (qubit1 == sqi) {
                if (isIPhaseFac) {
                    engine1->ApplySinglePhase(ZERO_CMPLX, I_CMPLX, sqi);
                    engine2->ApplySinglePhase(I_CMPLX, ZERO_CMPLX, sqi);
                }
                return;
            }

            if (isIPhaseFac) {
                engine1->ISwap(qubit1, sqi);
                engine2->ISwap(qubit1, sqi);
            } else {
                engine1->Swap(qubit1, sqi);
                engine2->Swap(qubit1, sqi);
            }

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
void QPager::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOp([&](QEnginePtr engine) { engine->FSim(theta, phi, qubit1, qubit2); }, { qubit1, qubit2 });
}

void QPager::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    // TODO: Get rid of this entirely. The need for this points to a bug in the general area of
    // ApplyAntiControlledSingleBit().

    bitLenInt qpp = qubitsPerPage();

    bitCapIntOcl i;

    if (start >= qpp) {
        // Entirely meta-
        start -= qpp;
        bitCapIntOcl mask = pow2Ocl(start + length) - pow2Ocl(start);
        for (i = 0; i < qPages.size(); i++) {
            if ((i & mask) == 0U) {
                qPages[i]->PhaseFlip();
            }
        }

        return;
    }

    if ((start + length) > qpp) {
        // Semi-meta-
        bitLenInt metaLen = (start + length) - qpp;
        bitLenInt remainderLen = length - metaLen;
        bitCapIntOcl mask = pow2Ocl(metaLen) - ONE_BCI;
        for (i = 0; i < qPages.size(); i++) {
            if ((i & mask) == 0U) {
                qPages[i]->ZeroPhaseFlip(start, remainderLen);
            }
        }

        return;
    }

    // Contained in sub-units
    for (i = 0; i < qPages.size(); i++) {
        qPages[i]->ZeroPhaseFlip(start, length);
    }
}

real1_f QPager::Prob(bitLenInt qubit)
{
    if (qPages.size() == 1U) {
        return qPages[0]->Prob(qubit);
    }

    real1 oneChance = ZERO_R1;
    bitCapIntOcl i;
    bitLenInt qpp = qubitsPerPage();
    std::vector<std::future<real1_f>> futures;

    if (qubit < qpp) {
        for (i = 0; i < qPages.size(); i++) {
            QEnginePtr engine = qPages[i];
            futures.push_back(std::async(std::launch::async, [engine, qubit]() { return engine->Prob(qubit); }));
        }
    } else {
        bitCapIntOcl qPower = pow2Ocl(qubit - qpp);
        bitCapIntOcl qMask = qPower - ONE_BCI;
        bitCapIntOcl fSize = qPages.size() >> ONE_BCI;
        bitCapIntOcl j;
        for (i = 0; i < fSize; i++) {
            j = i & qMask;
            j |= qPower | ((i ^ j) << ONE_BCI);

            QEnginePtr engine = qPages[j];
            futures.push_back(std::async(std::launch::async, [engine]() {
                engine->UpdateRunningNorm();
                return engine->GetRunningNorm();
            }));
        }
    }

    for (i = 0; i < futures.size(); i++) {
        oneChance += futures[i].get();
    }

    return oneChance;
}

real1_f QPager::ProbAll(bitCapInt fullRegister)
{
    bitCapIntOcl subIndex = (bitCapIntOcl)(fullRegister / pageMaxQPower());
    fullRegister -= subIndex * pageMaxQPower();
    return qPages[subIndex]->ProbAll(fullRegister);
}

real1_f QPager::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    CombineEngines(log2(mask));

    real1 maskChance = 0;
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        maskChance += qPages[i]->ProbMask(mask, permutation);
    }
    return maskChance;
}

bool QPager::ApproxCompare(QInterfacePtr toCompare, real1_f error_tol)
{
    QPagerPtr toComparePager = std::dynamic_pointer_cast<QPager>(toCompare);
    CombineEngines();
    toComparePager->CombineEngines();
    bool toRet = qPages[0]->ApproxCompare(toComparePager->qPages[0], error_tol);
    return toRet;
}

void QPager::UpdateRunningNorm(real1_f norm_thresh)
{
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->UpdateRunningNorm(norm_thresh);
    }
}

QInterfacePtr QPager::Clone()
{
    SeparateEngines();

    QPagerPtr clone = std::dynamic_pointer_cast<QPager>(
        CreateQuantumInterface(QINTERFACE_QPAGER, engine, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
            randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse));

    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        clone->qPages[i] = std::dynamic_pointer_cast<QEngine>(qPages[i]->Clone());
    }

    return clone;
}

} // namespace Qrack
