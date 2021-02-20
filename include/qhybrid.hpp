//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
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

class QHybrid;
typedef std::shared_ptr<QHybrid> QHybridPtr;

/**
 * A "Qrack::QHybrid" internally switched between Qrack::QEngineCPU and Qrack::QEngineOCL to maximize
 * qubit-count-dependent performance.
 */
class QHybrid : public QEngine {
protected:
    QEnginePtr engine;
    int devID;
    complex phaseFactor;
    bool useRDRAND;
    bool isSparse;
    uint32_t concurrency;
    bitLenInt thresholdQubits;
    bool isGpu;

public:
    QHybrid(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0);

    QEnginePtr MakeEngine(bool isOpenCL, bitCapInt initState = 0);

    virtual bool IsOpencl() { return isGpu; }

    virtual void SetConcurrency(uint32_t threadCount)
    {
        concurrency = threadCount;
        engine->SetConcurrency(concurrency);
    }

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    virtual void SwitchModes(bool useGpu)
    {
        if (!isGpu && useGpu) {
            QEnginePtr nEngine = MakeEngine(true);
            nEngine->CopyStateVec(engine);
            engine = nEngine;
        } else if (isGpu && !useGpu) {
            QEnginePtr nEngine = MakeEngine(false);
            nEngine->CopyStateVec(engine);
            engine = nEngine;
        }

        isGpu = useGpu;
    }

    virtual real1_f GetRunningNorm() { return engine->GetRunningNorm(); }

    virtual void ZeroAmplitudes() { engine->ZeroAmplitudes(); }

    virtual bool IsZeroAmplitude() { return engine->IsZeroAmplitude(); }

    virtual void CopyStateVec(QEnginePtr src) { CopyStateVec(std::dynamic_pointer_cast<QHybrid>(src)); }
    virtual void CopyStateVec(QHybridPtr src)
    {
        SwitchModes(src->isGpu);
        engine->CopyStateVec(src->engine);
    }

    virtual void GetAmplitudePage(complex* pagePtr, const bitCapInt offset, const bitCapInt length)
    {
        engine->GetAmplitudePage(pagePtr, offset, length);
    }
    virtual void SetAmplitudePage(const complex* pagePtr, const bitCapInt offset, const bitCapInt length)
    {
        engine->SetAmplitudePage(pagePtr, offset, length);
    }
    virtual void SetAmplitudePage(
        QHybridPtr pageEnginePtr, const bitCapInt srcOffset, const bitCapInt dstOffset, const bitCapInt length)
    {
        pageEnginePtr->SwitchModes(isGpu);
        engine->SetAmplitudePage(pageEnginePtr->engine, srcOffset, dstOffset, length);
    }
    virtual void SetAmplitudePage(
        QEnginePtr pageEnginePtr, const bitCapInt srcOffset, const bitCapInt dstOffset, const bitCapInt length)
    {
        SetAmplitudePage(std::dynamic_pointer_cast<QHybrid>(pageEnginePtr), srcOffset, dstOffset, length);
    }
    virtual void ShuffleBuffers(QEnginePtr oEngine) { ShuffleBuffers(std::dynamic_pointer_cast<QHybrid>(oEngine)); }
    virtual void ShuffleBuffers(QHybridPtr oEngine) { engine->ShuffleBuffers(oEngine->engine); }
    virtual void QueueSetDoNormalize(const bool& doNorm) { engine->QueueSetDoNormalize(doNorm); }
    virtual void QueueSetRunningNorm(const real1_f& runningNrm) { engine->QueueSetRunningNorm(runningNrm); }

    virtual void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm) { engine->ApplyM(regMask, result, nrm); }
    virtual real1_f ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
    {
        return engine->ProbReg(start, length, permutation);
    }

    using QInterface::Compose;
    virtual bitLenInt Compose(QHybridPtr toCopy)
    {
        bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;
        SwitchModes(nQubitCount >= thresholdQubits);
        toCopy->SwitchModes(isGpu);
        SetQubitCount(nQubitCount);
        return engine->Compose(toCopy->engine);
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QHybrid>(toCopy)); }
    virtual bitLenInt Compose(QHybridPtr toCopy, bitLenInt start)
    {
        bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;
        SwitchModes(nQubitCount >= thresholdQubits);
        toCopy->SwitchModes(isGpu);
        SetQubitCount(nQubitCount);
        return engine->Compose(toCopy->engine, start);
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QHybrid>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QHybrid>(dest));
    }
    virtual bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = REAL1_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QHybrid>(dest), error_tol);
    }
    virtual void Decompose(bitLenInt start, QHybridPtr dest)
    {
        bitLenInt nQubitCount = qubitCount - dest->GetQubitCount();
        SwitchModes(nQubitCount >= thresholdQubits);
        dest->SwitchModes(isGpu);
        SetQubitCount(nQubitCount);
        return engine->Decompose(start, dest->engine);
    }
    virtual void Dispose(bitLenInt start, bitLenInt length)
    {
        bitLenInt nQubitCount = qubitCount - length;
        SwitchModes(nQubitCount >= thresholdQubits);
        SetQubitCount(nQubitCount);
        return engine->Dispose(start, length);
    }
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        bitLenInt nQubitCount = qubitCount - length;
        SwitchModes(nQubitCount >= thresholdQubits);
        SetQubitCount(nQubitCount);
        return engine->Dispose(start, length, disposedPerm);
    }

    virtual bool TryDecompose(bitLenInt start, QHybridPtr dest, real1_f error_tol = REAL1_EPSILON)
    {
        bitLenInt nQubitCount = qubitCount - dest->GetQubitCount();
        SwitchModes(nQubitCount >= thresholdQubits);
        dest->SwitchModes(isGpu);
        bool result = engine->TryDecompose(start, dest->engine, error_tol);
        if (result) {
            SetQubitCount(nQubitCount);
        } else {
            SwitchModes(qubitCount >= thresholdQubits);
        }
        return result;
    }

    virtual void SetQuantumState(const complex* inputState) { engine->SetQuantumState(inputState); }
    virtual void GetQuantumState(complex* outputState) { engine->GetQuantumState(outputState); }
    virtual void GetProbs(real1* outputProbs) { engine->GetProbs(outputProbs); }
    virtual complex GetAmplitude(bitCapInt perm) { return engine->GetAmplitude(perm); }
    virtual void SetAmplitude(bitCapInt perm, complex amp) { engine->SetAmplitude(perm, amp); }
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        engine->SetPermutation(perm, phaseFac);
    }

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubitIndex) { engine->ApplySingleBit(mtrx, qubitIndex); }
    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex)
    {
        engine->ApplySinglePhase(topLeft, bottomRight, qubitIndex);
    }
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex)
    {
        engine->ApplySingleInvert(topRight, bottomLeft, qubitIndex);
    }
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        engine->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
    }
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        engine->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
    }
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask)
    {
        engine->UniformlyControlledSingleBit(
            controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
    }
    virtual void UniformParityRZ(const bitCapInt& mask, const real1_f& angle) { engine->UniformParityRZ(mask, angle); }
    virtual void CUniformParityRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle)
    {
        engine->CUniformParityRZ(controls, controlLen, mask, angle);
    }

    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        engine->CSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        engine->AntiCSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        engine->CSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        engine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        engine->CISqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        engine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
    }

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) { engine->INC(toAdd, start, length); }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        engine->CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCC(toAdd, start, length, carryIndex);
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        engine->INCS(toAdd, start, length, overflowIndex);
    }
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        engine->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCSC(toAdd, start, length, carryIndex);
    }
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) { engine->INCBCD(toAdd, start, length); }
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCBCDC(toAdd, start, length, carryIndex);
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECC(toSub, start, length, carryIndex);
    }
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        engine->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECSC(toSub, start, length, carryIndex);
    }
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECBCDC(toSub, start, length, carryIndex);
    }
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        engine->MUL(toMul, inOutStart, carryStart, length);
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        engine->DIV(toDiv, inOutStart, carryStart, length);
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->POWModNOut(base, modN, inStart, outStart, length);
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        engine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        engine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        engine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
    }

    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        engine->PhaseFlipIfLess(greaterPerm, start, length);
    }

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true)
    {
        return engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values) { engine->Hash(start, length, values); }

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->Swap(qubitIndex1, qubitIndex2); }
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->ISwap(qubitIndex1, qubitIndex2); }
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->SqrtSwap(qubitIndex1, qubitIndex2); }
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    virtual real1_f Prob(bitLenInt qubitIndex) { return engine->Prob(qubitIndex); }
    virtual real1_f ProbAll(bitCapInt fullRegister) { return engine->ProbAll(fullRegister); }
    virtual real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        return engine->ProbMask(mask, permutation);
    }
    virtual real1_f ProbParity(const bitCapInt& mask) { return engine->ProbParity(mask); }
    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true)
    {
        return engine->ForceMParity(mask, result, doForce);
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QHybrid>(toCompare));
    }
    virtual real1_f SumSqrDiff(QHybridPtr toCompare)
    {
        toCompare->SwitchModes(isGpu);
        return engine->SumSqrDiff(toCompare->engine);
    }

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) { engine->UpdateRunningNorm(norm_thresh); }
    virtual void NormalizeState(real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        engine->NormalizeState(nrm, norm_thresh);
    }

    virtual void Finish() { engine->Finish(); }

    virtual bool isFinished() { return engine->isFinished(); }

    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1, real1_f error_tol = REAL1_EPSILON)
    {
        return engine->TrySeparate(start, length, error_tol);
    }

    virtual QInterfacePtr Clone();

    virtual void SetDevice(const int& dID, const bool& forceReInit = false)
    {
        devID = dID;
        engine->SetDevice(dID, forceReInit);
    }

    virtual int GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize() { return engine->GetMaxSize(); };

protected:
    virtual real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
    {
        return engine->GetExpectation(valueStart, valueLength);
    }

    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        engine->Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }
    virtual void ApplyControlled2x2(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        engine->ApplyControlled2x2(controls, controlLen, target, mtrx);
    }
    virtual void ApplyAntiControlled2x2(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        engine->ApplyAntiControlled2x2(controls, controlLen, target, mtrx);
    }

    virtual void FreeStateVec(complex* sv = NULL) { engine->FreeStateVec(sv); }

    virtual void INCDECC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        engine->INCDECC(toMod, inOutStart, length, carryIndex);
    }
    virtual void INCDECSC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        engine->INCDECSC(toMod, inOutStart, length, carryIndex);
    }
    virtual void INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
        const bitLenInt& overflowIndex, const bitLenInt& carryIndex)
    {
        engine->INCDECSC(toMod, inOutStart, length, overflowIndex, carryIndex);
    }
    virtual void INCDECBCDC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        engine->INCDECBCDC(toMod, inOutStart, length, carryIndex);
    }
};
} // namespace Qrack
