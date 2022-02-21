//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#if ENABLE_PTHREAD
#include <future>
#endif
#include <string>

namespace Qrack {

QPager::QPager(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp,
    complex phaseFac, bool ignored, bool ignored2, bool useHostMem, int deviceId, bool useHardwareRNG,
    bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, false, useHardwareRNG, false, norm_thresh)
    , engines(eng)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , deviceIDs(devList)
    , useHardwareThreshold(false)
    , minPageQubits(0)
    , thresholdQubitsPerPage(qubitThreshold)
{
    if ((engines[0] == QINTERFACE_HYBRID) || (engines[0] == QINTERFACE_OPENCL)) {
#if ENABLE_OPENCL
        if (!OCLEngine::Instance().GetDeviceCount()) {
            engines[0] = QINTERFACE_CPU;
        }
#else
        engines[0] = QINTERFACE_CPU;
#endif
    }

#if ENABLE_ENV_VARS
    if (getenv("QRACK_QPAGER_DEVICES")) {
        std::string devListStr = std::string(getenv("QRACK_QPAGER_DEVICES"));
        deviceIDs.clear();
        if (devListStr.compare("") != 0) {
            std::stringstream devListStr_stream(devListStr);
            while (devListStr_stream.good()) {
                std::string substr;
                getline(devListStr_stream, substr, ',');
                deviceIDs.push_back(stoi(substr));
            }
        }
    }
#endif

    Init();

    if (!qubitCount) {
        return;
    }

    initState &= maxQPower - ONE_BCI;
    bitCapIntOcl pagePerm = 0;
    for (bitCapIntOcl i = 0; i < basePageCount; i++) {
        bool isPermInPage = (initState >= pagePerm);
        pagePerm += basePageMaxQPower;
        isPermInPage &= (initState < pagePerm);
        if (isPermInPage) {
            qPages.push_back(MakeEngine(
                baseQubitsPerPage, initState - (pagePerm - basePageMaxQPower), deviceIDs[i % deviceIDs.size()]));
        } else {
            qPages.push_back(MakeEngine(baseQubitsPerPage, 0, deviceIDs[i % deviceIDs.size()]));
            qPages.back()->ZeroAmplitudes();
        }
    }
}

QPager::QPager(QEnginePtr enginePtr, std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool ignored, bool ignored2, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, false, useHardwareRNG, false, norm_thresh)
    , engines(eng)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , deviceIDs(devList)
    , useHardwareThreshold(false)
    , segmentGlobalQb(0)
    , minPageQubits(0)
    , maxPageQubits(-1)
    , thresholdQubitsPerPage(qubitThreshold)
{
    Init();
    LockEngine(enginePtr);
}

void QPager::Init()
{
    if ((engines[0] == QINTERFACE_HYBRID) || (engines[0] == QINTERFACE_OPENCL)) {
#if ENABLE_OPENCL
        if (!OCLEngine::Instance().GetDeviceCount()) {
            engines[0] = QINTERFACE_CPU;
        }
#else
        engines[0] = QINTERFACE_CPU;
#endif
    }

#if ENABLE_ENV_VARS
    if (getenv("QRACK_SEGMENT_GLOBAL_QB")) {
        segmentGlobalQb = (bitLenInt)std::stoi(std::string(getenv("QRACK_SEGMENT_GLOBAL_QB")));
    }
#endif
    bitLenInt engineLevel = 0;
    rootEngine = engines[0];
    while ((engines.size() < engineLevel) && (rootEngine != QINTERFACE_CPU) && (rootEngine != QINTERFACE_OPENCL) &&
        (rootEngine != QINTERFACE_HYBRID)) {
        engineLevel++;
        rootEngine = engines[engineLevel];
    }

#if ENABLE_OPENCL
    if (rootEngine != QINTERFACE_CPU) {
        maxPageQubits =
            log2(OCLEngine::Instance().GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex)) - segmentGlobalQb;
    }

    if ((rootEngine != QINTERFACE_CPU) && (rootEngine != QINTERFACE_OPENCL)) {
        rootEngine = QINTERFACE_HYBRID;
    }

    if ((thresholdQubitsPerPage == 0) && ((rootEngine == QINTERFACE_OPENCL) || (rootEngine == QINTERFACE_HYBRID))) {
        useHardwareThreshold = true;
        useGpuThreshold = true;

        // Limit at the power of 2 less-than-or-equal-to a full max memory allocation segment, or choose with
        // environment variable.
        thresholdQubitsPerPage = maxPageQubits;
    }
#endif

    if (thresholdQubitsPerPage == 0) {
        useHardwareThreshold = true;
        useGpuThreshold = false;

#if ENABLE_ENV_VARS
        const bitLenInt pStridePow =
            (bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW);
#else
        const bitLenInt pStridePow = PSTRIDEPOW;
#endif

#if ENABLE_PTHREAD
        const unsigned numCores = GetConcurrencyLevel();
        minPageQubits = pStridePow + ((numCores == 1U) ? 1U : (log2(numCores - 1U) + 1U));
#else
        minPageQubits = pStridePow + 1U;
#endif

        thresholdQubitsPerPage = minPageQubits;
    }

    if (deviceIDs.size() == 0) {
        deviceIDs.push_back(devID);
    }

    SetQubitCount(qubitCount);

    maxQubits = sizeof(bitCapIntOcl) * bitsInByte;
    if (baseQubitsPerPage > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");
    }
#if ENABLE_ENV_VARS
    if (getenv("QRACK_MAX_PAGING_QB")) {
        maxQubits = (bitLenInt)std::stoi(std::string(getenv("QRACK_MAX_PAGING_QB")));
    }
#endif
    if (qubitCount > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a QPager with greater capacity than environment variable QRACK_MAX_PAGING_QB.");
    }
}

QEnginePtr QPager::MakeEngine(bitLenInt length, bitCapInt perm, int deviceId)
{
    return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engines, length, perm, rand_generator, phaseFactor,
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

    const bitCapIntOcl groupCount = pow2Ocl(qubitCount - bit);
    const bitCapIntOcl groupSize = (bitCapIntOcl)(qPages.size() / groupCount);
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    std::vector<QEnginePtr> nQPages;

    for (bitCapIntOcl i = 0; i < groupCount; i++) {
        nQPages.push_back(MakeEngine(bit, 0, deviceIDs[i % deviceIDs.size()]));
        QEnginePtr engine = nQPages.back();
        for (bitCapIntOcl j = 0; j < groupSize; j++) {
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

    const bitCapIntOcl pagesPer = pow2Ocl(qubitCount - thresholdBits) / qPages.size();
    const bitCapIntOcl pageMaxQPower = pow2Ocl(thresholdBits);

    std::vector<QEnginePtr> nQPages;
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        for (bitCapIntOcl j = 0; j < pagesPer; j++) {
            nQPages.push_back(MakeEngine(thresholdBits, 0, deviceIDs[(j + (i * pagesPer)) % deviceIDs.size()]));
            nQPages.back()->SetAmplitudePage(qPages[i], j * pageMaxQPower, 0, pageMaxQPower);
        }
    }

    qPages = nQPages;
}

template <typename Qubit1Fn> void QPager::SingleBitGate(bitLenInt target, Qubit1Fn fn, bool isSqiCtrl, bool isAnti)
{
    bitLenInt qpp = qubitsPerPage();

    if (doNormalize) {
        real1_f runningNorm = ZERO_R1;
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Finish();
            runningNorm += qPages[i]->GetRunningNorm();
        }
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->QueueSetRunningNorm(runningNorm);
            qPages[i]->QueueSetDoNormalize(true);
        }
    }

    if (target < qpp) {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            QEnginePtr engine = qPages[i];
            fn(engine, target);
            engine->QueueSetDoNormalize(false);
        }

        return;
    }

    const bitLenInt sqi = qpp - 1U;
    target -= qpp;
    const bitCapIntOcl targetPow = pow2Ocl(target);
    const bitCapIntOcl targetMask = targetPow - ONE_BCI;
    const bitCapIntOcl maxLCV = (bitCapIntOcl)qPages.size() >> ONE_BCI;
#if ENABLE_PTHREAD
    std::vector<std::future<void>> futures(maxLCV);
#endif
    for (bitCapIntOcl i = 0; i < maxLCV; i++) {
        bitCapIntOcl j = i & targetMask;
        j |= (i ^ j) << ONE_BCI;

        QEnginePtr engine1 = qPages[j];
        QEnginePtr engine2 = qPages[j + targetPow];

        const bool doNorm = doNormalize;

#if ENABLE_PTHREAD
        futures[i] = std::async(std::launch::async, [engine1, engine2, fn, doNorm, sqi, isSqiCtrl, isAnti]() {
#endif
            engine1->ShuffleBuffers(engine2);

            if (!isSqiCtrl || isAnti) {
                fn(engine1, sqi);
            }

            if (!isSqiCtrl || !isAnti) {
                fn(engine2, sqi);
            }

            if (doNorm) {
                engine1->QueueSetDoNormalize(false);
                engine2->QueueSetDoNormalize(false);
            }

            engine1->ShuffleBuffers(engine2);
#if ENABLE_PTHREAD
        });
#endif
    }
#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
#endif
}

// This is like the QEngineCPU and QEngineOCL logic for register-like CNOT and CCNOT, just swapping sub-engine indices
// instead of amplitude indices.
template <typename Qubit1Fn>
void QPager::MetaControlled(bool anti, const std::vector<bitLenInt>& controls, bitLenInt target, Qubit1Fn fn,
    const complex* mtrx, bool isSqiCtrl, bool isIntraCtrled)
{
    const bitLenInt qpp = qubitsPerPage();
    const bitLenInt sqi = qpp - 1U;
    target -= qpp;

    std::vector<bitCapIntOcl> sortedMasks(1U + controls.size());
    const bitCapIntOcl targetPow = pow2Ocl(target);
    sortedMasks[controls.size()] = targetPow - ONE_BCI;

    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < (bitLenInt)controls.size(); i++) {
        sortedMasks[i] = pow2Ocl(controls[i] - qpp);
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bool isSpecial, isInvert;
    complex top, bottom;
    if (!isIntraCtrled && !isSqiCtrl && IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        isSpecial = true;
        isInvert = false;
        top = mtrx[0];
        bottom = mtrx[3];
    } else if (!isIntraCtrled && !isSqiCtrl && IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
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

    const bitCapIntOcl maxLCV = (bitCapIntOcl)qPages.size() >> (bitCapIntOcl)sortedMasks.size();
#if ENABLE_PTHREAD
    std::vector<std::future<void>> futures;
#endif
    for (bitCapIntOcl i = 0; i < maxLCV; i++) {
        bitCapIntOcl jHi = i;
        bitCapIntOcl j = 0;
        for (bitCapIntOcl k = 0; k < (sortedMasks.size()); k++) {
            bitCapIntOcl jLo = jHi & sortedMasks[k];
            jHi = (jHi ^ jLo) << ONE_BCI;
            j |= jLo;
        }
        j |= jHi | controlMask;

        if (isInvert) {
            qPages[j].swap(qPages[j + targetPow]);
        }

        QEnginePtr engine1 = qPages[j];
        QEnginePtr engine2 = qPages[j + targetPow];
        bool doTop, doBottom;

        if (isSpecial) {
            doTop = !IS_NORM_0(top);
            doBottom = !IS_NORM_0(bottom);

            if (doTop) {
                engine1->Phase(top, top, 0);
            }
            if (doBottom) {
                engine2->Phase(bottom, bottom, 0);
            }

            continue;
        }

        doTop = !isSqiCtrl || anti;
        doBottom = !isSqiCtrl || !anti;

#if ENABLE_PTHREAD
        futures.push_back(std::async(std::launch::async, [engine1, engine2, fn, sqi, doTop, doBottom]() {
#endif
            engine1->ShuffleBuffers(engine2);
            if (doTop) {
                fn(engine1, sqi);
            }
            if (doBottom) {
                fn(engine2, sqi);
            }
            engine1->ShuffleBuffers(engine2);
#if ENABLE_PTHREAD
        }));
#endif
    }

#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < futures.size(); i++) {
        futures[i].get();
    }
#endif
}

// This is called when control bits are "meta-" but the target bit is below the "meta-" threshold, (low enough to fit in
// sub-engines).
template <typename Qubit1Fn>
void QPager::SemiMetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn)
{
    const bitLenInt qpp = qubitsPerPage();

    std::vector<bitLenInt> sortedMasks(controls.size());

    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < (bitLenInt)controls.size(); i++) {
        sortedMasks[i] = pow2Ocl(controls[i] - qpp);
        if (!anti) {
            controlMask |= sortedMasks[i];
        }
        sortedMasks[i]--;
    }
    std::sort(sortedMasks.begin(), sortedMasks.end());

    const bitCapIntOcl maxLCV = (bitCapIntOcl)qPages.size() >> (bitCapIntOcl)sortedMasks.size();
    for (bitCapIntOcl i = 0; i < maxLCV; i++) {
        bitCapIntOcl jHi = i;
        bitCapIntOcl j = 0;
        for (bitCapIntOcl k = 0; k < (sortedMasks.size()); k++) {
            bitCapIntOcl jLo = jHi & sortedMasks[k];
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
    for (bitLenInt i = 0; i < (bitLenInt)bits.size(); i++) {
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

    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
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
    if ((qubitCount + toCopy->qubitCount) > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a QPager with greater capacity than environment variable QRACK_MAX_PAGING_QB.");
    }

    bitLenInt qpp = qubitsPerPage();
    bitLenInt tcqpp = toCopy->qubitsPerPage();

    if ((qpp + tcqpp) > maxPageQubits) {
        tcqpp = (maxPageQubits <= qpp) ? 1U : (maxPageQubits - qpp);
        toCopy->SeparateEngines(tcqpp, true);
    }

    if ((qpp + tcqpp) > maxPageQubits) {
        qpp = (maxPageQubits <= tcqpp) ? 1U : (maxPageQubits - tcqpp);
        SeparateEngines(qpp, true);
    }

    const bitLenInt pqc = pagedQubitCount();
    const bitCapIntOcl maxJ = ((bitCapIntOcl)toCopy->qPages.size() - 1U);
    std::vector<QEnginePtr> nQPages;
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
        for (bitCapIntOcl j = 0; j < maxJ; j++) {
            nQPages.push_back(std::dynamic_pointer_cast<QEngine>(engine->Clone()));
            nQPages.back()->Compose(toCopy->qPages[j]);
        }
        nQPages.push_back(engine);
        nQPages.back()->Compose(toCopy->qPages[maxJ]);
    }

    qPages = nQPages;

    bitLenInt toRet = qubitCount;
    SetQubitCount(qubitCount + toCopy->qubitCount);

    ROL(pqc, qpp, pqc + toCopy->qubitCount);

    return toRet;
}

bitLenInt QPager::Compose(QPagerPtr toCopy, bitLenInt start)
{
    if (start == qubitCount) {
        return Compose(toCopy);
    }

    if ((qubitCount + toCopy->qubitCount) > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a QPager with greater capacity than environment variable QRACK_MAX_PAGING_QB.");
    }

    // TODO: Avoid CombineEngines();
    CombineEngines();
    toCopy->CombineEngines();

    qPages[0]->Compose(toCopy->qPages[0]);

    SetQubitCount(qubitCount + toCopy->qubitCount);

    return start;
}

void QPager::Decompose(bitLenInt start, QPagerPtr dest)
{
    const bitLenInt length = dest->qubitCount;
    if (start == 0) {
        CombineEngines(length + 1U);
    } else {
        CombineEngines(start + length);
    }
    dest->CombineEngines();
    bool didDecompose = false;
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        if (qPages[i]->GetRunningNorm() < ZERO_R1) {
            qPages[i]->UpdateRunningNorm();
        }

        if (didDecompose || (qPages[i]->GetRunningNorm() <= ZERO_R1)) {
            qPages[i]->Dispose(start, length);
        } else {
            qPages[i]->Decompose(start, dest->qPages[0]);
            didDecompose = true;
        }
    }

    SetQubitCount(qubitCount - length);
    CombineEngines(baseQubitsPerPage);
}

void QPager::Dispose(bitLenInt start, bitLenInt length)
{
    if (start == 0) {
        CombineEngines(length + 1U);
    } else {
        CombineEngines(start + length);
    }
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->Dispose(start, length);
    }

    SetQubitCount(qubitCount - length);
    CombineEngines(baseQubitsPerPage);
}

void QPager::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    if (start == 0) {
        CombineEngines(length + 1U);
    } else {
        CombineEngines(start + length);
    }
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->Dispose(start, length, disposedPerm);
    }

    SetQubitCount(qubitCount - length);
    CombineEngines(baseQubitsPerPage);
}

void QPager::SetQuantumState(const complex* inputState)
{
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    bitCapIntOcl pagePerm = 0;
#if ENABLE_PTHREAD
    std::vector<std::future<void>> futures(qPages.size());
#endif
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
        const bool doNorm = doNormalize;
#if ENABLE_PTHREAD
        futures[i] = std::async(std::launch::async, [engine, inputState, pagePerm, doNorm]() {
#endif
            engine->SetQuantumState(inputState + pagePerm);
            if (doNorm) {
                engine->UpdateRunningNorm();
            }
#if ENABLE_PTHREAD
        });
#endif
        pagePerm += pagePower;
    }

#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        futures[i].get();
    }
#endif
}

void QPager::GetQuantumState(complex* outputState)
{
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    bitCapIntOcl pagePerm = 0;
#if ENABLE_PTHREAD
    std::vector<std::future<void>> futures(qPages.size());
#endif
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
        futures[i] = std::async(
            std::launch::async, [engine, outputState, pagePerm]() { engine->GetQuantumState(outputState + pagePerm); });
#else
        engine->GetQuantumState(outputState + pagePerm);
#endif
        pagePerm += pagePower;
    }
#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        futures[i].get();
    }
#endif
}

void QPager::GetProbs(real1* outputProbs)
{
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    bitCapIntOcl pagePerm = 0;
#if ENABLE_PTHREAD
    std::vector<std::future<void>> futures(qPages.size());
#endif
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
        futures[i] = std::async(
            std::launch::async, [engine, outputProbs, pagePerm]() { engine->GetProbs(outputProbs + pagePerm); });
#else
        engine->GetProbs(outputProbs + pagePerm);
#endif
        pagePerm += pagePower;
    }
#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        futures[i].get();
    }
#endif
}

void QPager::SetPermutation(bitCapInt perm, complex phaseFac)
{
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    perm &= maxQPower - ONE_BCI;
    bitCapIntOcl pagePerm = 0;
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        bool isPermInPage = (perm >= pagePerm);
        pagePerm += pagePower;
        isPermInPage &= (perm < pagePerm);

        if (isPermInPage) {
            qPages[i]->SetPermutation(perm - (pagePerm - pagePower), phaseFac);
            continue;
        }

        qPages[i]->ZeroAmplitudes();
    }
}

void QPager::Mtrx(const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        Phase(mtrx[0], mtrx[3], target);
        return;
    } else if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        Invert(mtrx[1], mtrx[2], target);
        return;
    }

    SeparateEngines();
    SingleBitGate(target, [mtrx](QEnginePtr engine, bitLenInt lTarget) { engine->Mtrx(mtrx, lTarget); });
}

void QPager::ApplySingleEither(bool isInvert, complex top, complex bottom, bitLenInt target)
{
    SeparateEngines();
    bitLenInt qpp = qubitsPerPage();

    if (target < qpp) {
        if (isInvert) {
            SingleBitGate(
                target, [top, bottom](QEnginePtr engine, bitLenInt lTarget) { engine->Invert(top, bottom, lTarget); });
        } else {
            SingleBitGate(
                target, [top, bottom](QEnginePtr engine, bitLenInt lTarget) { engine->Phase(top, bottom, lTarget); });
        }

        return;
    }

    if (randGlobalPhase) {
        bottom /= top;
        top = ONE_CMPLX;
    }

    target -= qpp;
    const bitCapIntOcl targetPow = pow2Ocl(target);
    const bitCapIntOcl qMask = targetPow - 1U;
    const bitCapIntOcl maxLCV = (bitCapIntOcl)qPages.size() >> 1U;
    for (bitCapIntOcl i = 0; i < maxLCV; i++) {
        bitCapIntOcl j = i & qMask;
        j |= (i ^ j) << ONE_BCI;

        if (isInvert) {
            qPages[j].swap(qPages[j + targetPow]);
        }

        if (top != ONE_CMPLX) {
            qPages[j]->Phase(top, top, 0);
        }
        if (bottom != ONE_CMPLX) {
            qPages[j + targetPow]->Phase(bottom, bottom, 0);
        }
    }
}

void QPager::ApplyEitherControlledSingleBit(
    bool anti, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, const complex* mtrx)
{
    if (controlLen == 0) {
        Mtrx(mtrx, target);
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
                engine->MACMtrx(&(intraControls[0]), intraControls.size(), mtrx, lTarget);
            } else {
                engine->MCMtrx(&(intraControls[0]), intraControls.size(), mtrx, lTarget);
            }
        } else {
            engine->Mtrx(mtrx, lTarget);
        }
    };

    if (metaControls.size() == 0) {
        SingleBitGate(target, sg, isSqiCtrl, anti);
    } else if (target < qpp) {
        SemiMetaControlled(anti, metaControls, target, sg);
    } else {
        MetaControlled(anti, metaControls, target, sg, mtrx, isSqiCtrl, intraControls.size() > 0);
    }
}

void QPager::UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask)
{
    CombineAndOpControlled(
        [&](QEnginePtr engine) {
            engine->UniformlyControlledSingleBit(
                controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
        },
        { qubitIndex }, controls, controlLen);
}

void QPager::UniformParityRZ(bitCapInt mask, real1_f angle)
{
    CombineAndOp([&](QEnginePtr engine) { engine->UniformParityRZ(mask, angle); }, { log2(mask) });
}

void QPager::CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
{
    CombineAndOpControlled([&](QEnginePtr engine) { engine->CUniformParityRZ(controls, controlLen, mask, angle); },
        { log2(mask) }, controls, controlLen);
}

void QPager::CSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
{
    if (controlLen == 0) {
        Swap(qubit1, qubit2);
        return;
    }

    if (qubit1 == qubit2) {
        return;
    }

    CombineAndOpControlled([&](QEnginePtr engine) { engine->CSwap(controls, controlLen, qubit1, qubit2); },
        { qubit1, qubit2 }, controls, controlLen);
}
void QPager::AntiCSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
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
void QPager::CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
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
void QPager::AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
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
void QPager::CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
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
void QPager::AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
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

void QPager::XMask(bitCapInt mask)
{
    const bitCapInt pageMask = pageMaxQPower() - ONE_BCI;
    const bitCapIntOcl intraMask = (bitCapIntOcl)(mask & pageMask);
    bitCapInt interMask = mask ^ (bitCapInt)intraMask;
    bitCapInt v;
    bitLenInt bit;
    while (interMask) {
        v = interMask & (interMask - ONE_BCI);
        bit = log2(interMask ^ v);
        interMask = v;
        X(bit);
    }

    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->XMask(intraMask);
    }
}

void QPager::PhaseParity(real1_f radians, bitCapInt mask)
{
    const bitCapIntOcl parityStartSize = 4U * sizeof(bitCapIntOcl);
    const bitCapInt pageMask = pageMaxQPower() - ONE_BCI;
    const bitCapIntOcl intraMask = (bitCapIntOcl)(mask & pageMask);
    const bitCapIntOcl interMask = (((bitCapIntOcl)mask) ^ intraMask) >> qubitsPerPage();
    const complex phaseFac = std::polar(ONE_R1, (real1)(radians / 2));
    const complex iPhaseFac = ONE_CMPLX / phaseFac;
    bitCapIntOcl v;
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];

        v = interMask & i;
        for (bitCapIntOcl paritySize = parityStartSize; paritySize > 0U; paritySize >>= 1U) {
            v ^= v >> paritySize;
        }
        v &= 1U;

        if (intraMask) {
            engine->PhaseParity(v ? -radians : radians, intraMask);
        } else if (v) {
            engine->Phase(phaseFac, phaseFac, 0U);
        } else {
            engine->Phase(iPhaseFac, iPhaseFac, 0U);
        }
    }
}

bool QPager::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (qPages.size() == 1U) {
        return qPages[0]->ForceM(qubit, result, doForce, doApply);
    }

    real1_f oneChance = Prob(qubit);
    if (!doForce) {
        if (oneChance >= ONE_R1) {
            result = true;
        } else if (oneChance <= ZERO_R1) {
            result = false;
        } else {
            real1_f prob = Rand();
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
        throw std::invalid_argument("ERROR: Forced a measurement result with 0 probability");
    }

    if (doApply && (nrmlzr != ONE_BCI)) {
        const bitLenInt qpp = qubitsPerPage();
#if ENABLE_PTHREAD
        std::vector<std::future<void>> futures(qPages.size());
#endif
        if (qubit < qpp) {
            const complex nrmFac = GetNonunitaryPhase() / (real1)std::sqrt(nrmlzr);
            const bitCapIntOcl qPower = pow2Ocl(qubit);
            for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
                QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
                futures[i] = (std::async(std::launch::async,
                    [engine, qPower, result, nrmFac]() { engine->ApplyM(qPower, result, nrmFac); }));
#else
                engine->ApplyM(qPower, result, nrmFac);
#endif
            }
        } else {
            const bitLenInt metaQubit = qubit - qpp;
            const bitCapIntOcl qPower = pow2Ocl(metaQubit);
            for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
                QEnginePtr engine = qPages[i];
                if (!(i & qPower) == !result) {
#if ENABLE_PTHREAD
                    futures[i] =
                        (std::async(std::launch::async, [engine, nrmlzr]() { engine->NormalizeState(nrmlzr); }));
#else
                    engine->NormalizeState(nrmlzr);
#endif
                } else {
#if ENABLE_PTHREAD
                    futures[i] = (std::async(std::launch::async, [engine]() { engine->ZeroAmplitudes(); }));
#else
                    engine->ZeroAmplitudes();
#endif
                }
            }
        }

#if ENABLE_PTHREAD
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            futures[i].get();
        }
#endif
    }

    return result;
}

#if ENABLE_ALU
void QPager::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCS(toAdd, start, length, overflowIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex });
}
void QPager::INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCDECSC(toAdd, start, length, overflowIndex, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), overflowIndex, carryIndex });
}
void QPager::INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCDECSC(toAdd, start, length, carryIndex); },
        { static_cast<bitLenInt>(start + length - 1U), carryIndex });
}
#if ENABLE_BCD
void QPager::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCBCD(toAdd, start, length); },
        { static_cast<bitLenInt>(start + length - 1U) });
}
void QPager::INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    CombineAndOp([&](QEnginePtr engine) { engine->INCDECBCDC(toAdd, start, length, carryIndex); },
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
void QPager::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
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
void QPager::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
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
    const bitLenInt* controls, bitLenInt controlLen)
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
    const bitLenInt* controls, bitLenInt controlLen)
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
    const bitLenInt* controls, bitLenInt controlLen)
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

bitCapInt QPager::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    const unsigned char* values, bool ignored)
{
    CombineAndOp(
        [&](QEnginePtr engine) { engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, true); },
        { static_cast<bitLenInt>(indexStart + indexLength - 1U),
            static_cast<bitLenInt>(valueStart + valueLength - 1U) });

    return 0;
}

bitCapInt QPager::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
    bitLenInt carryIndex, const unsigned char* values)
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
    bitLenInt carryIndex, const unsigned char* values)
{
    CombineAndOp(
        [&](QEnginePtr engine) {
            engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        },
        { static_cast<bitLenInt>(indexStart + indexLength - 1U), static_cast<bitLenInt>(valueStart + valueLength - 1U),
            carryIndex });

    return 0;
}
void QPager::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    CombineAndOp([&](QEnginePtr engine) { engine->Hash(start, length, values); },
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
#endif

void QPager::MetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac)
{
    const bitLenInt qpp = qubitsPerPage();
    qubit1 -= qpp;
    qubit2 -= qpp;

    std::vector<bitCapIntOcl> sortedMasks(2U);
    const bitCapIntOcl qubit1Pow = pow2Ocl(qubit1);
    sortedMasks[0] = qubit1Pow - ONE_BCI;
    const bitCapIntOcl qubit2Pow = pow2Ocl(qubit2);
    sortedMasks[1] = qubit2Pow - ONE_BCI;
    std::sort(sortedMasks.begin(), sortedMasks.end());

    bitCapIntOcl maxLCV = (bitCapIntOcl)qPages.size() >> (bitCapIntOcl)sortedMasks.size();
    for (bitCapIntOcl i = 0; i < maxLCV; i++) {
        bitCapIntOcl j = i & sortedMasks[0];
        bitCapIntOcl jHi = (i ^ j) << ONE_BCI;
        bitCapIntOcl jLo = jHi & sortedMasks[1];
        j |= jLo | ((jHi ^ jLo) << ONE_BCI);

        qPages[j + qubit1Pow].swap(qPages[j + qubit2Pow]);

        if (!isIPhaseFac) {
            continue;
        }

        qPages[j + qubit1Pow]->Phase(I_CMPLX, I_CMPLX, 0);
        qPages[j + qubit2Pow]->Phase(I_CMPLX, I_CMPLX, 0);
    }
}

void QPager::SemiMetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac)
{
    if (qubit1 > qubit2) {
        std::swap(qubit1, qubit2);
    }

    const bitLenInt qpp = qubitsPerPage();
    const bitLenInt sqi = qpp - 1U;
    qubit2 -= qpp;

    const bitCapIntOcl qubit2Pow = pow2Ocl(qubit2);
    const bitCapIntOcl qubit2Mask = qubit2Pow - ONE_BCI;
    const bitCapIntOcl maxLCV = (bitCapIntOcl)qPages.size() >> ONE_BCI;
#if ENABLE_PTHREAD
    std::vector<std::future<void>> futures(maxLCV);
#endif
    for (bitCapIntOcl i = 0; i < maxLCV; i++) {
        bitCapIntOcl j = i & qubit2Mask;
        j |= (i ^ j) << ONE_BCI;

        QEnginePtr engine1 = qPages[j];
        QEnginePtr engine2 = qPages[j + qubit2Pow];

#if ENABLE_PTHREAD
        futures[i] = std::async(std::launch::async, [engine1, engine2, qubit1, isIPhaseFac, sqi]() {
#endif
            engine1->ShuffleBuffers(engine2);

            if (qubit1 == sqi) {
                if (isIPhaseFac) {
                    engine1->Phase(ZERO_CMPLX, I_CMPLX, sqi);
                    engine2->Phase(I_CMPLX, ZERO_CMPLX, sqi);
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
#if ENABLE_PTHREAD
        });
#endif
    }

#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < maxLCV; i++) {
        futures[i].get();
    }
#endif
}

void QPager::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const bool isQubit1Meta = qubit1 >= baseQubitsPerPage;
    const bool isQubit2Meta = qubit2 >= baseQubitsPerPage;
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

    const bool isQubit1Meta = qubit1 >= baseQubitsPerPage;
    const bool isQubit2Meta = qubit2 >= baseQubitsPerPage;
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

real1_f QPager::Prob(bitLenInt qubit)
{
    if (qPages.size() == 1U) {
        return qPages[0]->Prob(qubit);
    }

    const bitLenInt qpp = qubitsPerPage();
    real1 oneChance = ZERO_R1;
#if ENABLE_PTHREAD
    std::vector<std::future<real1_f>> futures;
#endif

    if (qubit < qpp) {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
            futures.push_back(std::async(std::launch::async, [engine, qubit]() { return engine->Prob(qubit); }));
#else
            oneChance += engine->Prob(qubit);
#endif
        }
    } else {
        const bitCapIntOcl qPower = pow2Ocl(qubit - qpp);
        const bitCapIntOcl qMask = qPower - ONE_BCI;
        const bitCapIntOcl fSize = (bitCapIntOcl)qPages.size() >> ONE_BCI;
        for (bitCapIntOcl i = 0; i < fSize; i++) {
            bitCapIntOcl j = i & qMask;
            j |= ((i ^ j) << ONE_BCI) | qPower;

            QEnginePtr engine = qPages[j];
#if ENABLE_PTHREAD
            futures.push_back(std::async(std::launch::async, [engine]() {
                engine->UpdateRunningNorm();
                return engine->GetRunningNorm();
            }));
#else
            engine->UpdateRunningNorm();
            oneChance += engine->GetRunningNorm();
#endif
        }
    }

#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < futures.size(); i++) {
        oneChance += futures[i].get();
    }
#endif

    return clampProb(oneChance);
}

real1_f QPager::ProbMask(bitCapInt mask, bitCapInt permutation)
{
    CombineEngines(log2(mask));

    real1_f maskChance = ZERO_R1;
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        maskChance += qPages[i]->ProbMask(mask, permutation);
    }
    return clampProb(maskChance);
}

real1_f QPager::ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset)
{
    if (length != qubitCount) {
        return QInterface::ExpectationBitsAll(bits, length, offset);
    }

    for (bitCapIntOcl i = 0; i < length; i++) {
        if (bits[i] != i) {
            return QInterface::ExpectationBitsAll(bits, length, offset);
        }
    }

    const bitLenInt qpp = qubitsPerPage();
    const bitCapIntOcl pagePower = (bitCapIntOcl)pageMaxQPower();
    real1_f expectation = ZERO_R1;
    bitCapIntOcl pagePerm = 0;
#if ENABLE_PTHREAD
    std::vector<std::future<real1_f>> futures(qPages.size());
#endif
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        QEnginePtr engine = qPages[i];
#if ENABLE_PTHREAD
        futures[i] = std::async(std::launch::async, [engine, bits, qpp, pagePerm, offset]() {
            return engine->ExpectationBitsAll(bits, qpp, pagePerm + offset);
        });
#else
        expectation += engine->ExpectationBitsAll(bits, qpp, pagePerm + offset);
#endif
        pagePerm += pagePower;
    }
#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        expectation += futures[i].get();
    }
#endif

    return expectation;
}

void QPager::UpdateRunningNorm(real1_f norm_thresh)
{
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->UpdateRunningNorm(norm_thresh);
    }
}

void QPager::NormalizeState(real1_f nrm, real1_f norm_thresh, real1_f phaseArg)
{
    real1_f nmlzr;
    if (nrm == REAL1_DEFAULT_ARG) {
        nmlzr = ZERO_R1;
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            nmlzr += qPages[i]->GetRunningNorm();
        }
    } else {
        nmlzr = nrm;
    }

    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        qPages[i]->NormalizeState(nmlzr, norm_thresh, phaseArg);
    }
}

QInterfacePtr QPager::Clone()
{
    SeparateEngines();

    QPagerPtr clone = std::make_shared<QPager>(engines, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse, (real1_f)amplitudeFloor);

    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        clone->qPages[i] = std::dynamic_pointer_cast<QEngine>(qPages[i]->Clone());
    }

    return clone;
}

real1_f QPager::SumSqrDiff(QPagerPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1;
    }

    SeparateEngines(toCompare->qubitsPerPage());
    toCompare->SeparateEngines(qubitsPerPage());
    CombineEngines(toCompare->qubitsPerPage());
    toCompare->CombineEngines(qubitsPerPage());

    real1_f toRet = ZERO_R1;
#if ENABLE_PTHREAD
    std::vector<std::future<real1_f>> futures(qPages.size());
#endif
    for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
        QEnginePtr lEngine = qPages[i];
        QEnginePtr rEngine = toCompare->qPages[i];
#if ENABLE_PTHREAD
        futures[i] = (std::async(std::launch::async, [lEngine, rEngine]() { return lEngine->SumSqrDiff(rEngine); }));
#else
        toRet += lEngine->SumSqrDiff(rEngine);
#endif
    }

#if ENABLE_PTHREAD
    for (bitCapIntOcl i = 0; i < futures.size(); i++) {
        toRet += futures[i].get();
    }
#endif

    return toRet;
}

} // namespace Qrack
