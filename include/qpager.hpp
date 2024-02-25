//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.
#pragma once

#include "qengine.hpp"
#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif
#if ENABLE_CUDA
#include "common/cudaengine.cuh"
#endif

namespace Qrack {

class QPager;
typedef std::shared_ptr<QPager> QPagerPtr;

/**
 * A "Qrack::QPager" splits a "Qrack::QEngine" implementation into equal-length "pages." This helps both optimization
 * and distribution of a single coherent quantum register across multiple devices.
 */
class QPager : public QEngine, public std::enable_shared_from_this<QPager> {
protected:
    bool useGpuThreshold;
    bool isSparse;
    bool useTGadget;
    bitLenInt maxPageSetting;
    bitLenInt maxPageQubits;
    bitLenInt thresholdQubitsPerPage;
    bitLenInt baseQubitsPerPage;
    bitLenInt maxQubits;
    int64_t devID;
    QInterfaceEngine rootEngine;
    bitCapIntOcl basePageMaxQPower;
    bitCapIntOcl basePageCount;
    complex phaseFactor;
    std::vector<bool> devicesHostPointer;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engines;
    std::vector<QEnginePtr> qPages;

    QEnginePtr MakeEngine(bitLenInt length, bitCapIntOcl pageId);

    void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        baseQubitsPerPage = (qubitCount < thresholdQubitsPerPage) ? qubitCount : thresholdQubitsPerPage;
        basePageCount = pow2Ocl(qubitCount - baseQubitsPerPage);
        basePageMaxQPower = pow2Ocl(baseQubitsPerPage);
    }

    bitCapIntOcl pageMaxQPower()
    {
        bitCapInt toRet;
        bi_div_mod_small(maxQPower, qPages.size(), &toRet, NULL);
        return (bitCapIntOcl)toRet;
    }
    bitLenInt pagedQubitCount() { return log2Ocl(qPages.size()); }
    bitLenInt qubitsPerPage() { return log2Ocl(pageMaxQPower()); }
    int64_t GetPageDevice(bitCapIntOcl page) { return deviceIDs[page % deviceIDs.size()]; }
    bool GetPageHostPointer(bitCapIntOcl page) { return devicesHostPointer[page % devicesHostPointer.size()]; }

    void CombineEngines(bitLenInt thresholdBits);
    void CombineEngines() { CombineEngines(qubitCount); }
    void SeparateEngines(bitLenInt thresholdBits, bool noBaseFloor = false);
    void SeparateEngines() { SeparateEngines(baseQubitsPerPage); }

    template <typename Qubit1Fn>
    void SingleBitGate(bitLenInt target, Qubit1Fn fn, bool isSqiCtrl = false, bool isAnti = false);
    template <typename Qubit1Fn>
    void MetaControlled(bitCapInt controlPerm, const std::vector<bitLenInt>& controls, bitLenInt target, Qubit1Fn fn,
        const complex* mtrx, bool isSqiCtrl = false, bool isIntraCtrled = false);
    template <typename Qubit1Fn>
    void SemiMetaControlled(bitCapInt controlPerm, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn);
    void MetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac, bool isInverse);

    template <typename F> void CombineAndOp(F fn, std::vector<bitLenInt> bits);
    template <typename F>
    void CombineAndOpControlled(F fn, std::vector<bitLenInt> bits, const std::vector<bitLenInt>& controls);

    void ApplySingleEither(bool isInvert, complex top, complex bottom, bitLenInt target);
    void ApplyEitherControlledSingleBit(
        bitCapInt controlPerm, const std::vector<bitLenInt>& controls, bitLenInt target, const complex* mtrx);
    void EitherISwap(bitLenInt qubit1, bitLenInt qubit2, bool isInverse);

    void Init();

    void GetSetAmplitudePage(complex* pagePtr, const complex* cPagePtr, bitCapIntOcl offset, bitCapIntOcl length);

public:
    QPager(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool ignored = false, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QPager(bitLenInt qBitCount, bitCapInt initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool ignored = false, bool useHostMem = false,
        int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
#if ENABLE_OPENCL
        : QPager({ OCLEngine::Instance().GetDeviceCount() ? QINTERFACE_OPENCL : QINTERFACE_CPU }, qBitCount, initState,
              rgp, phaseFac, doNorm, ignored, useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh,
              devList, qubitThreshold, separation_thresh)
#elif ENABLE_CUDA
        : QPager({ CUDAEngine::Instance().GetDeviceCount() ? QINTERFACE_CUDA : QINTERFACE_CPU }, qBitCount, initState,
              rgp, phaseFac, doNorm, ignored, useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh,
              devList, qubitThreshold, separation_thresh)
#else
        : QPager({ QINTERFACE_CPU }, qBitCount, initState, rgp, phaseFac, doNorm, ignored, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold, separation_thresh)
#endif
    {
    }

    QPager(QEnginePtr enginePtr, std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt ignored = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool ignored2 = false, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    void SetConcurrency(uint32_t threadsPerEngine)
    {
        QInterface::SetConcurrency(threadsPerEngine);
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->SetConcurrency(threadsPerEngine);
        }
    }
    void SetTInjection(bool useGadget)
    {
        useTGadget = useGadget;
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->SetTInjection(useTGadget);
        }
    }
    bool GetTInjection() { return useTGadget; }
    bool isOpenCL() { return qPages[0U]->isOpenCL(); }

    QEnginePtr ReleaseEngine()
    {
        CombineEngines();
        return qPages[0U];
    }

    void LockEngine(QEnginePtr eng)
    {
        qPages.resize(1U);
        qPages[0U] = eng;
        eng->SetDevice(deviceIDs[0]);
        SeparateEngines();
    }

    void ZeroAmplitudes()
    {
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->ZeroAmplitudes();
        }
    }
    void CopyStateVec(QEnginePtr src) { CopyStateVec(std::dynamic_pointer_cast<QPager>(src)); }
    void CopyStateVec(QPagerPtr src)
    {
        bitLenInt qpp = qubitsPerPage();
        src->CombineEngines(qpp);
        src->SeparateEngines(qpp, true);

        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->CopyStateVec(src->qPages[i]);
        }
    }
    bool IsZeroAmplitude()
    {
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            if (!qPages[i]->IsZeroAmplitude()) {
                return false;
            }
        }

        return true;
    }
    void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        GetSetAmplitudePage(pagePtr, NULL, offset, length);
    }
    void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        GetSetAmplitudePage(NULL, pagePtr, offset, length);
    }
    void SetAmplitudePage(QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        SetAmplitudePage(std::dynamic_pointer_cast<QPager>(pageEnginePtr), srcOffset, dstOffset, length);
    }
    void SetAmplitudePage(QPagerPtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        CombineEngines();
        pageEnginePtr->CombineEngines();
        qPages[0U]->SetAmplitudePage(pageEnginePtr->qPages[0U], srcOffset, dstOffset, length);
    }
    void ShuffleBuffers(QEnginePtr engine) { ShuffleBuffers(std::dynamic_pointer_cast<QPager>(engine)); }
    void ShuffleBuffers(QPagerPtr engine)
    {
        bitLenInt qpp = qubitsPerPage();
        bitLenInt tcqpp = engine->qubitsPerPage();
        engine->SeparateEngines(qpp, true);
        SeparateEngines(tcqpp, true);

        if (qPages.size() == 1U) {
            qPages[0U]->ShuffleBuffers(engine->qPages[0U]);
            return;
        }

        const bitCapIntOcl offset = qPages.size() >> 1U;
        for (bitCapIntOcl i = 0U; i < offset; ++i) {
            qPages[offset + i].swap(engine->qPages[i]);
        }
    }
    QEnginePtr CloneEmpty();
    void QueueSetDoNormalize(bool doNorm)
    {
        Finish();
        doNormalize = doNorm;
    }
    void QueueSetRunningNorm(real1_f runningNrm)
    {
        Finish();
        runningNorm = runningNrm;
    }
    real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
    {
        CombineEngines();
        return qPages[0U]->ProbReg(start, length, permutation);
    }
    using QEngine::ApplyM;
    void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
    {
        CombineEngines();
        return qPages[0U]->ApplyM(regMask, result, nrm);
    }
    real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
    {
        CombineEngines();
        return qPages[0U]->GetExpectation(valueStart, valueLength);
    }
    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        CombineEngines();
        qPages[0U]->Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }
    real1_f GetRunningNorm()
    {
        real1_f toRet = ZERO_R1_F;
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            toRet += qPages[i]->GetRunningNorm();
        }

        return toRet;
    }

    real1_f FirstNonzeroPhase()
    {
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            if (!qPages[i]->IsZeroAmplitude()) {
                return qPages[i]->FirstNonzeroPhase();
            }
        }

        return ZERO_R1_F;
    }

    void SetQuantumState(const complex* inputState);
    void GetQuantumState(complex* outputState);
    void GetProbs(real1* outputProbs);
    complex GetAmplitude(bitCapInt perm)
    {
        bitCapInt p, a;
        bi_div_mod(perm, pageMaxQPower(), &p, &a);
        return qPages[(bitCapIntOcl)p]->GetAmplitude(a);
    }
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        bitCapInt p, a;
        bi_div_mod(perm, pageMaxQPower(), &p, &a);
        qPages[(bitCapIntOcl)p]->SetAmplitude(a, amp);
    }
    real1_f ProbAll(bitCapInt perm)
    {
        bitCapInt p, a;
        bi_div_mod(perm, pageMaxQPower(), &p, &a);
        return qPages[(bitCapIntOcl)p]->ProbAll(a);
    }

    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

    using QEngine::Compose;
    bitLenInt Compose(QPagerPtr toCopy) { return ComposeEither(toCopy, false); }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QPager>(toCopy)); }
    bitLenInt ComposeNoClone(QPagerPtr toCopy) { return ComposeEither(toCopy, true); }
    bitLenInt ComposeNoClone(QInterfacePtr toCopy) { return ComposeNoClone(std::dynamic_pointer_cast<QPager>(toCopy)); }
    bitLenInt ComposeEither(QPagerPtr toCopy, bool willDestroy);
    void Decompose(bitLenInt start, QInterfacePtr dest) { Decompose(start, std::dynamic_pointer_cast<QPager>(dest)); }
    void Decompose(bitLenInt start, QPagerPtr dest);
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);
    using QEngine::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length);

    void Mtrx(const complex* mtrx, bitLenInt target);
    void Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex)
    {
        ApplySingleEither(false, topLeft, bottomRight, qubitIndex);
    }
    void Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex)
    {
        ApplySingleEither(true, topRight, bottomLeft, qubitIndex);
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        bitCapInt p = pow2(controls.size());
        bi_decrement(&p, 1U);
        ApplyEitherControlledSingleBit(p, controls, target, mtrx);
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        ApplyEitherControlledSingleBit(ZERO_BCI, controls, target, mtrx);
    }

    void UniformParityRZ(bitCapInt mask, real1_f angle);
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, bitCapInt mask, real1_f angle);

    void XMask(bitCapInt mask);
    void ZMask(bitCapInt mask) { PhaseParity((real1_f)PI_R1, mask); }
    void PhaseParity(real1_f radians, bitCapInt mask);

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);
    bitCapInt ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true, bool doApply = true)
    {
        // Don't use QEngine::ForceMReg().
        return QInterface::ForceMReg(start, length, result, doForce, doApply);
    }

#if ENABLE_ALU
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls);

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true);
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values);
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values);
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values);

    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
#endif

    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    void ISwap(bitLenInt qubit1, bitLenInt qubit2) { EitherISwap(qubit1, qubit2, false); }
    void IISwap(bitLenInt qubit1, bitLenInt qubit2) { EitherISwap(qubit1, qubit2, true); }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    real1_f Prob(bitLenInt qubitIndex);
    real1_f ProbMask(bitCapInt mask, bitCapInt permutation);
    // TODO: QPager not yet used in Q#, but this would need a real implementation:
    real1_f ProbParity(bitCapInt mask)
    {
        if (bi_compare_0(mask) == 0) {
            return ZERO_R1_F;
        }

        CombineEngines();
        return qPages[0U]->ProbParity(mask);
    }
    bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
    {
        if (bi_compare_0(mask) == 0) {
            return ZERO_R1_F;
        }

        CombineEngines();
        return qPages[0U]->ForceMParity(mask, result, doForce);
    }
    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, bitCapInt offset = ZERO_BCI);

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);

    void Finish()
    {
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->Finish();
        }
    };

    bool isFinished()
    {
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            if (!qPages[i]->isFinished()) {
                return false;
            }
        }

        return true;
    };

    void Dump()
    {
        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->Dump();
        }
    };

    QInterfacePtr Clone();

    void SetDevice(int64_t dID)
    {
        deviceIDs.clear();
        deviceIDs.push_back(dID);

        for (bitCapIntOcl i = 0U; i < qPages.size(); ++i) {
            qPages[i]->SetDevice(dID);
        }

#if ENABLE_OPENCL || ENABLE_CUDA
        if (rootEngine != QINTERFACE_CPU) {
#if ENABLE_OPENCL
            maxPageQubits =
                log2Ocl(OCLEngine::Instance().GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex)) - 1U;
#else
            maxPageQubits =
                log2Ocl(CUDAEngine::Instance().GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex)) - 1U;
#endif
            if (maxPageSetting < maxPageQubits) {
                maxPageQubits = maxPageSetting;
            }
        }

        if (!useGpuThreshold) {
            return;
        }

        // Limit at the power of 2 less-than-or-equal-to a full max memory allocation segment, or choose with
        // environment variable.
        thresholdQubitsPerPage = maxPageQubits;
#endif
    }

    int64_t GetDevice() { return qPages[0U]->GetDevice(); }

    bitCapIntOcl GetMaxSize() { return qPages[0U]->GetMaxSize(); };

    real1_f SumSqrDiff(QInterfacePtr toCompare) { return SumSqrDiff(std::dynamic_pointer_cast<QPager>(toCompare)); }

    real1_f SumSqrDiff(QPagerPtr toCompare);
};
} // namespace Qrack
