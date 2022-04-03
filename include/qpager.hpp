//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
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

namespace Qrack {

class QPager;
typedef std::shared_ptr<QPager> QPagerPtr;

/**
 * A "Qrack::QPager" splits a "Qrack::QEngine" implementation into equal-length "pages." This helps both optimization
 * and distribution of a single coherent quantum register across multiple devices.
 */
class QPager : public QEngine {
protected:
    std::vector<QInterfaceEngine> engines;
    QInterfaceEngine rootEngine;
    int devID;
    complex phaseFactor;
    bool isSparse;
    std::vector<QEnginePtr> qPages;
    std::vector<int> deviceIDs;

    bool useHardwareThreshold;
    bool useGpuThreshold;
    bitLenInt segmentGlobalQb;
    bitLenInt minPageQubits;
    bitLenInt maxPageQubits;
    bitLenInt thresholdQubitsPerPage;
    bitLenInt baseQubitsPerPage;
    bitCapInt basePageCount;
    bitCapIntOcl basePageMaxQPower;

    bitLenInt maxQubits;

    QEnginePtr MakeEngine(bitLenInt length, bitCapInt perm, int deviceId);

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        baseQubitsPerPage = (qubitCount < thresholdQubitsPerPage) ? qubitCount : thresholdQubitsPerPage;
        basePageCount = pow2Ocl(qubitCount - baseQubitsPerPage);
        basePageMaxQPower = pow2Ocl(baseQubitsPerPage);
    }

    bitCapInt pageMaxQPower() { return maxQPower / qPages.size(); }
    bitLenInt pagedQubitCount() { return log2((bitCapInt)qPages.size()); }
    bitLenInt qubitsPerPage() { return log2(pageMaxQPower()); }

    void CombineEngines(bitLenInt thresholdBits);
    void CombineEngines() { CombineEngines(qubitCount); }
    void SeparateEngines(bitLenInt thresholdBits, bool noBaseFloor = false);
    void SeparateEngines() { SeparateEngines(baseQubitsPerPage); }

    template <typename Qubit1Fn>
    void SingleBitGate(bitLenInt target, Qubit1Fn fn, bool isSqiCtrl = false, bool isAnti = false);
    template <typename Qubit1Fn>
    void MetaControlled(bool anti, const std::vector<bitLenInt>& controls, bitLenInt target, Qubit1Fn fn,
        const complex* mtrx, bool isSqiCtrl = false, bool isIntraCtrled = false);
    template <typename Qubit1Fn>
    void SemiMetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn);
    void MetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac);
    void SemiMetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac);

    template <typename F> void CombineAndOp(F fn, std::vector<bitLenInt> bits);
    template <typename F>
    void CombineAndOpControlled(F fn, std::vector<bitLenInt> bits, const bitLenInt* controls, bitLenInt controlLen);

    void ApplySingleEither(bool isInvert, complex top, complex bottom, bitLenInt target);
    void ApplyEitherControlledSingleBit(
        bool anti, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, const complex* mtrx);

    void Init();

    virtual void GetSetAmplitudePage(
        complex* pagePtr, const complex* cPagePtr, bitCapIntOcl offset, bitCapIntOcl length);

#if ENABLE_OPENCL
    virtual size_t GetRequiredSpace(bitCapIntOcl pageQubits);
#endif

public:
    QPager(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool ignored = false, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QPager(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool ignored = false, bool useHostMem = false,
        int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QPager({ QINTERFACE_MASK_FUSION }, qBitCount, initState, rgp, phaseFac, doNorm, ignored, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold, separation_thresh)
    {
    }

    QPager(QEnginePtr enginePtr, std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt ignored = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool ignored2 = false, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON_F);

    virtual void SetConcurrency(uint32_t threadsPerEngine)
    {
        QInterface::SetConcurrency(threadsPerEngine);
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->SetConcurrency(threadsPerEngine);
        }
    }

    virtual QEnginePtr ReleaseEngine()
    {
        CombineEngines();
        return qPages[0];
    }

    virtual void LockEngine(QEnginePtr eng)
    {
        qPages.resize(1);
        qPages[0] = eng;
    }

    virtual void ZeroAmplitudes()
    {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->ZeroAmplitudes();
        }
    }
    virtual void CopyStateVec(QEnginePtr src) { CopyStateVec(std::dynamic_pointer_cast<QPager>(src)); }
    virtual void CopyStateVec(QPagerPtr src)
    {
        bitLenInt qpp = qubitsPerPage();
        src->CombineEngines(qpp);
        src->SeparateEngines(qpp, true);

        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->CopyStateVec(src->qPages[i]);
        }
    }
    virtual bool IsZeroAmplitude()
    {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            if (!qPages[i]->IsZeroAmplitude()) {
                return false;
            }
        }

        return true;
    }
    virtual void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        GetSetAmplitudePage(pagePtr, NULL, offset, length);
    }
    virtual void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        GetSetAmplitudePage(NULL, pagePtr, offset, length);
    }
    virtual void SetAmplitudePage(
        QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        SetAmplitudePage(std::dynamic_pointer_cast<QPager>(pageEnginePtr), srcOffset, dstOffset, length);
    }
    virtual void SetAmplitudePage(
        QPagerPtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        CombineEngines();
        pageEnginePtr->CombineEngines();
        qPages[0]->SetAmplitudePage(pageEnginePtr->qPages[0], srcOffset, dstOffset, length);
    }
    virtual void ShuffleBuffers(QEnginePtr engine) { ShuffleBuffers(std::dynamic_pointer_cast<QPager>(engine)); }
    virtual void ShuffleBuffers(QPagerPtr engine)
    {
        bitLenInt qpp = qubitsPerPage();
        bitLenInt tcqpp = engine->qubitsPerPage();
        engine->SeparateEngines(qpp, true);
        SeparateEngines(tcqpp, true);

        if (qPages.size() == 1U) {
            qPages[0]->ShuffleBuffers(engine->qPages[0]);
            return;
        }

        const bitCapIntOcl offset = qPages.size() >> 1U;
        for (bitCapIntOcl i = 0U; i < offset; i++) {
            qPages[offset + i].swap(engine->qPages[i]);
        }
    }
    virtual QEnginePtr CloneEmpty();
    virtual void QueueSetDoNormalize(bool doNorm)
    {
        Finish();
        doNormalize = doNorm;
    }
    virtual void QueueSetRunningNorm(real1_f runningNrm)
    {
        Finish();
        runningNorm = runningNrm;
    }
    virtual real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
    {
        CombineEngines();
        return qPages[0]->ProbReg(start, length, permutation);
    }
    virtual void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
    {
        CombineEngines();
        return qPages[0]->ApplyM(regMask, result, nrm);
    }
    virtual real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
    {
        CombineEngines();
        return qPages[0]->GetExpectation(valueStart, valueLength);
    }
    virtual void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        CombineEngines();
        qPages[0]->Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }
    virtual void FreeStateVec(complex* sv = NULL)
    {
        CombineEngines();
        qPages[0]->FreeStateVec(sv);
    }
    virtual real1_f GetRunningNorm()
    {
        real1_f toRet = ZERO_R1_F;
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            toRet += qPages[i]->GetRunningNorm();
        }

        return toRet;
    }

    virtual real1_f FirstNonzeroPhase()
    {
        for (bitCapIntOcl i = 0U; i < qPages.size(); i++) {
            if (!qPages[i]->IsZeroAmplitude()) {
                return qPages[i]->FirstNonzeroPhase();
            }
        }

        return ZERO_R1_F;
    }

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm)
    {
        bitCapIntOcl subIndex = (bitCapIntOcl)(perm / pageMaxQPower());
        return qPages[subIndex]->GetAmplitude(perm & (pageMaxQPower() - ONE_BCI));
    }
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        bitCapIntOcl subIndex = (bitCapIntOcl)(perm / pageMaxQPower());
        qPages[subIndex]->SetAmplitude(perm & (pageMaxQPower() - ONE_BCI), amp);
    }
    real1_f ProbAll(bitCapInt fullRegister)
    {
        bitCapIntOcl subIndex = (bitCapIntOcl)(fullRegister / pageMaxQPower());
        return qPages[subIndex]->ProbAll(fullRegister & (pageMaxQPower() - ONE_BCI));
    }

    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

    using QEngine::Compose;
    virtual bitLenInt Compose(QPagerPtr toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QPager>(toCopy)); }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QPager>(dest));
    }
    virtual void Decompose(bitLenInt start, QPagerPtr dest);
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    virtual void Mtrx(const complex* mtrx, bitLenInt target);
    virtual void Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex)
    {
        ApplySingleEither(false, topLeft, bottomRight, qubitIndex);
    }
    virtual void Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex)
    {
        ApplySingleEither(true, topRight, bottomLeft, qubitIndex);
    }
    virtual void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
    {
        ApplyEitherControlledSingleBit(false, controls, controlLen, target, mtrx);
    }
    virtual void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
    {
        ApplyEitherControlledSingleBit(true, controls, controlLen, target, mtrx);
    }
    virtual void UniformParityRZ(bitCapInt mask, real1_f angle);
    virtual void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle);

    virtual void XMask(bitCapInt mask);
    virtual void ZMask(bitCapInt mask) { PhaseParity((real1_f)PI_R1, mask); }
    virtual void PhaseParity(real1_f radians, bitCapInt mask);

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true, bool doApply = true)
    {
        // Don't use QEngine::ForceMReg().
        return QInterface::ForceMReg(start, length, result, doForce, doApply);
    }

#if ENABLE_ALU
    virtual void INCDECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, const unsigned char* values);

    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
#endif

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    virtual real1_f Prob(bitLenInt qubitIndex);
    virtual real1_f ProbMask(bitCapInt mask, bitCapInt permutation);
    // TODO: QPager not yet used in Q#, but this would need a real implementation:
    virtual real1_f ProbParity(bitCapInt mask)
    {
        if (!mask) {
            return ZERO_R1_F;
        }

        CombineEngines();
        return qPages[0]->ProbParity(mask);
    }
    virtual bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
    {
        if (!mask) {
            return ZERO_R1_F;
        }

        CombineEngines();
        return qPages[0]->ForceMParity(mask, result, doForce);
    }
    virtual real1_f ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset = 0);

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);

    virtual void Finish()
    {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Finish();
        }
    };

    virtual bool isFinished()
    {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            if (!qPages[i]->isFinished()) {
                return false;
            }
        }

        return true;
    };

    virtual void Dump()
    {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Dump();
        }
    };

    virtual QInterfacePtr Clone();

    virtual void SetDevice(int dID, bool forceReInit = false)
    {
        deviceIDs.clear();
        deviceIDs.push_back(dID);

        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->SetDevice(dID, forceReInit);
        }

#if ENABLE_OPENCL
        if (rootEngine != QINTERFACE_CPU) {
            maxPageQubits = log2(OCLEngine::Instance().GetDeviceContextPtr(devID)->GetMaxAlloc() / sizeof(complex)) -
                segmentGlobalQb;
        }

        if (!useGpuThreshold) {
            return;
        }

        // Limit at the power of 2 less-than-or-equal-to a full max memory allocation segment, or choose with
        // environment variable.
        thresholdQubitsPerPage = maxPageQubits;
#endif
    }

    virtual int64_t GetDevice() { return qPages[0]->GetDevice(); }

    bitCapIntOcl GetMaxSize() { return qPages[0]->GetMaxSize(); };

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QPager>(toCompare));
    }

    virtual real1_f SumSqrDiff(QPagerPtr toCompare);
};
} // namespace Qrack
