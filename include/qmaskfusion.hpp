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

#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif

#include "qengine.hpp"

namespace Qrack {

struct QMaskFusionShard {
    bool isX;
    bool isZ;
    uint64_t phase;
    bool isBuffered() { return isX || isZ || phase; }

    QMaskFusionShard()
        : isX(false)
        , isZ(false)
        , phase(0)
    {
    }
};

class QMaskFusion;
typedef std::shared_ptr<QMaskFusion> QMaskFusionPtr;

/**
 * A "Qrack::QMaskFusion" internally switched between Qrack::QEngineCPU and Qrack::QEngineOCL to maximize
 * qubit-count-dependent performance.
 */
class QMaskFusion : public QEngine {
protected:
    bool isCacheEmpty;
    bool useRDRAND;
    bool isSparse;
    bitLenInt thresholdQubits;
    real1_f separabilityThreshold;
    int64_t devID;
    complex phaseFactor;
    QEnginePtr engine;
    std::vector<int64_t> devices;
    std::vector<QInterfaceEngine> engTypes;
    std::vector<QMaskFusionShard> zxShards;

    QEnginePtr MakeEngine(bitCapInt initState = 0);

    void FlushBuffers();
    void DumpBuffers()
    {
        isCacheEmpty = true;
        DumpBuffers(0, qubitCount);
    }

    void DumpBuffers(bitLenInt start, bitLenInt length)
    {
        bitLenInt maxLcv = start + length;
        for (bitLenInt i = start; i < maxLcv; ++i) {
            DumpBuffer(i);
        }
    }

    void DumpBuffer(bitLenInt target)
    {
        zxShards[target].isX = false;
        zxShards[target].isZ = false;
    }

    bool FlushIfBuffered(bitLenInt target)
    {
        if (isCacheEmpty) {
            return true;
        }

        if (zxShards[target].isBuffered()) {
            FlushBuffers();
            return true;
        }

        return false;
    }

    bool FlushIfBuffered(bitLenInt start, bitLenInt length)
    {
        if (isCacheEmpty) {
            return true;
        }

        bitLenInt maxLcv = start + length;
        for (bitLenInt i = start; i < maxLcv; ++i) {
            if (zxShards[i].isBuffered()) {
                FlushBuffers();
                return true;
            }
        }

        return false;
    }

    bool FlushIfPhaseBlocked() { return FlushIfPhaseBlocked((bitLenInt)0U, qubitCount); }

    bool FlushIfPhaseBlocked(const bitLenInt* controls, bitLenInt controlLen)
    {
        if (isCacheEmpty) {
            return true;
        }

        bool isBlocked = false;
        for (bitLenInt i = 0U; i < controlLen; ++i) {
            bitLenInt control = controls[i];
            isBlocked = zxShards[control].isX;
            if (isBlocked) {
                break;
            }
        }

        if (isBlocked) {
            FlushBuffers();
        }

        return isBlocked;
    }

    bool FlushIfPhaseBlocked(bitLenInt target)
    {
        if (isCacheEmpty) {
            return true;
        }

        bool isBlocked = zxShards[target].isX;
        if (isBlocked) {
            FlushBuffers();
        }

        return isBlocked;
    }

    bool FlushIfPhaseBlocked(bitLenInt start, bitLenInt length)
    {
        if (isCacheEmpty) {
            return true;
        }

        bool isBlocked = false;
        bitLenInt maxLcv = start + length;
        for (bitLenInt i = start; i < maxLcv; ++i) {
            isBlocked = zxShards[i].isX;
            if (isBlocked) {
                break;
            }
        }
        if (isBlocked) {
            FlushBuffers();
        }

        return isBlocked;
    }

public:
    QMaskFusion(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0U,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);
    QMaskFusion(bitLenInt qBitCount, bitCapInt initState = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QMaskFusion({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    void ZeroAmplitudes()
    {
        DumpBuffers();
        engine->ZeroAmplitudes();
    }
    bool IsZeroAmplitude() { return engine->IsZeroAmplitude(); }
    void CopyStateVec(QEnginePtr src) { CopyStateVec(std::dynamic_pointer_cast<QMaskFusion>(src)); }
    void CopyStateVec(QMaskFusionPtr src)
    {
        FlushBuffers();
        engine->CopyStateVec(src->engine);
    }
    void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        FlushBuffers();
        engine->GetAmplitudePage(pagePtr, offset, length);
    }
    void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        FlushBuffers();
        engine->SetAmplitudePage(pagePtr, offset, length);
    }
    void SetAmplitudePage(QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        SetAmplitudePage(std::dynamic_pointer_cast<QMaskFusion>(pageEnginePtr), srcOffset, dstOffset, length);
    }
    void SetAmplitudePage(
        QMaskFusionPtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        FlushBuffers();
        pageEnginePtr->FlushBuffers();
        engine->SetAmplitudePage(pageEnginePtr->engine, srcOffset, dstOffset, length);
    }
    void ShuffleBuffers(QEnginePtr oEngine) { ShuffleBuffers(std::dynamic_pointer_cast<QMaskFusion>(oEngine)); }
    void ShuffleBuffers(QMaskFusionPtr oEngine)
    {
        FlushBuffers();
        oEngine->FlushBuffers();
        engine->ShuffleBuffers(oEngine->engine);
    }
    QEnginePtr CloneEmpty() { return engine->CloneEmpty(); }
    void QueueSetDoNormalize(bool doNorm) { engine->QueueSetDoNormalize(doNorm); }
    void QueueSetRunningNorm(real1_f runningNrm) { engine->QueueSetRunningNorm(runningNrm); }
    real1_f GetRunningNorm() { return engine->GetRunningNorm(); }

    real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
    {
        FlushIfPhaseBlocked(start, length);
        return engine->ProbReg(start, length, permutation);
    }

    using QEngine::Compose;
    bitLenInt Compose(QMaskFusionPtr toCopy)
    {
        bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;
        isCacheEmpty = false;
        zxShards.insert(zxShards.end(), toCopy->zxShards.begin(), toCopy->zxShards.end());
        SetQubitCount(nQubitCount);
        return engine->Compose(toCopy->engine);
    }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QMaskFusion>(toCopy)); }
    bitLenInt Compose(QMaskFusionPtr toCopy, bitLenInt start)
    {
        bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;
        isCacheEmpty = false;
        zxShards.insert(zxShards.begin() + start, toCopy->zxShards.begin(), toCopy->zxShards.end());
        SetQubitCount(nQubitCount);
        return engine->Compose(toCopy->engine, start);
    }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QMaskFusion>(toCopy), start);
    }
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QMaskFusion>(dest));
    }
    bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QMaskFusion>(dest), error_tol);
    }
    using QEngine::Decompose;
    void Decompose(bitLenInt start, QMaskFusionPtr dest)
    {
        bitLenInt length = dest->GetQubitCount();
        bitLenInt nQubitCount = qubitCount - length;
        std::copy(zxShards.begin() + start, zxShards.begin() + start + length, dest->zxShards.begin());
        zxShards.erase(zxShards.begin() + start, zxShards.begin() + start + length);
        SetQubitCount(nQubitCount);
        return engine->Decompose(start, dest->engine);
    }
    void Dispose(bitLenInt start, bitLenInt length)
    {
        bitLenInt nQubitCount = qubitCount - length;
        zxShards.erase(zxShards.begin() + start, zxShards.begin() + start + length);
        SetQubitCount(nQubitCount);
        return engine->Dispose(start, length);
    }
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        FlushBuffers();
        bitLenInt nQubitCount = qubitCount - length;
        zxShards.erase(zxShards.begin() + start, zxShards.begin() + start + length);
        SetQubitCount(nQubitCount);
        return engine->Dispose(start, length, disposedPerm);
    }
    bool TryDecompose(bitLenInt start, QMaskFusionPtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        bitLenInt length = dest->GetQubitCount();
        bitLenInt nQubitCount = qubitCount - length;
        bool result = engine->TryDecompose(start, dest->engine, error_tol);
        if (result) {
            std::copy(zxShards.begin() + start, zxShards.begin() + start + length, dest->zxShards.begin());
            zxShards.erase(zxShards.begin() + start, zxShards.begin() + start + length);
            SetQubitCount(nQubitCount);
        }
        return result;
    }

    void SetQuantumState(const complex* inputState)
    {
        DumpBuffers();
        engine->SetQuantumState(inputState);
    }
    void GetQuantumState(complex* outputState)
    {
        FlushBuffers();
        engine->GetQuantumState(outputState);
    }
    void GetProbs(real1* outputProbs)
    {
        FlushIfPhaseBlocked();
        engine->GetProbs(outputProbs);
    }
    complex GetAmplitude(bitCapInt perm)
    {
        FlushBuffers();
        return engine->GetAmplitude(perm);
    }
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        FlushBuffers();
        engine->SetAmplitude(perm, amp);
    }
    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        DumpBuffers();
        engine->SetPermutation(perm, phaseFac);
    }

    using QEngine::X;
    void X(bitLenInt target)
    {
        QMaskFusionShard& shard = zxShards[target];
        shard.isX = !shard.isX;
        isCacheEmpty = false;
    }
    using QEngine::Y;
    void Y(bitLenInt target)
    {
        Z(target);
        X(target);
        QMaskFusionShard& shard = zxShards[target];
        if (!randGlobalPhase) {
            shard.phase = (shard.phase + 1U) & 3U;
        }
    }
    using QEngine::Z;
    void Z(bitLenInt target)
    {
        QMaskFusionShard& shard = zxShards[target];
        if (!randGlobalPhase && shard.isX) {
            shard.phase = (shard.phase + 2U) & 3U;
        }
        shard.isZ = !shard.isZ;
        isCacheEmpty = false;
    }

    void Mtrx(const complex* mtrx, bitLenInt target);
    void Phase(complex topLeft, complex bottomRight, bitLenInt target);
    void Invert(complex topRight, complex bottomLeft, bitLenInt target);

    void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
    {
        if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
            MCPhase(controls, controlLen, mtrx[0U], mtrx[3U], target);
            return;
        }

        FlushIfBuffered(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->MCMtrx(controls, controlLen, mtrx, target);
    }
    void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
    {
        if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
            MACPhase(controls, controlLen, mtrx[0U], mtrx[3U], target);
            return;
        }

        FlushIfBuffered(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->MACMtrx(controls, controlLen, mtrx, target);
    }
    void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target)
    {
        FlushIfPhaseBlocked(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->MCPhase(controls, controlLen, topLeft, bottomRight, target);
    }
    void MACPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target)
    {
        FlushIfPhaseBlocked(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->MACPhase(controls, controlLen, topLeft, bottomRight, target);
    }

    using QEngine::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
        const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask)
    {
        FlushIfBuffered(qubitIndex) || FlushIfPhaseBlocked(controls, controlLen);
        engine->UniformlyControlledSingleBit(
            controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
    }

    void UniformParityRZ(bitCapInt mask, real1_f angle) { engine->UniformParityRZ(mask, angle); }
    void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
    {
        FlushBuffers();
        engine->CUniformParityRZ(controls, controlLen, mask, angle);
    }

    void CSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->CSwap(controls, controlLen, qubit1, qubit2);
    }
    void AntiCSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->AntiCSwap(controls, controlLen, qubit1, qubit2);
    }
    void CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->CSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    void AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    void CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->CISqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    void AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
    }

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        FlushIfPhaseBlocked(qubit);
        DumpBuffer(qubit);
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    using QEngine::ApplyM;
    void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm) { engine->ApplyM(regMask, result, nrm); }

#if ENABLE_ALU
    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        FlushIfBuffered(start, length);
        engine->INC(toAdd, start, length);
    }
    void CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfPhaseBlocked(controls, controlLen);
        engine->CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(overflowIndex);
        engine->INCS(toAdd, start, length, overflowIndex);
    }
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        FlushIfBuffered(start, length);
        engine->INCBCD(toAdd, start, length);
    }
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryStart, length);
        engine->MUL(toMul, inOutStart, carryStart, length);
    }
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryStart, length);
        engine->DIV(toDiv, inOutStart, carryStart, length);
    }
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length);
        engine->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length);
        engine->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length);
        engine->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        FlushIfBuffered(indexStart, indexLength) || FlushIfBuffered(valueStart, valueLength);
        return engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        FlushIfBuffered(indexStart, indexLength) || FlushIfBuffered(valueStart, valueLength) ||
            FlushIfBuffered(carryIndex);
        return engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        FlushIfBuffered(indexStart, indexLength) || FlushIfBuffered(valueStart, valueLength) ||
            FlushIfBuffered(carryIndex);
        return engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        FlushIfBuffered(start, length);
        engine->Hash(start, length, values);
    }

    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(flagIndex);
        engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        FlushIfBuffered(start, length);
        engine->PhaseFlipIfLess(greaterPerm, start, length);
    }
#endif

    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        std::swap(zxShards[qubitIndex1], zxShards[qubitIndex2]);
        engine->Swap(qubitIndex1, qubitIndex2);
    }
    void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->ISwap(qubitIndex1, qubitIndex2);
    }
    void IISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->IISwap(qubitIndex1, qubitIndex2);
    }
    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
    }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    real1_f Prob(bitLenInt qubitIndex)
    {
        if (zxShards[qubitIndex].isX) {
            return clampProb(ONE_R1_F - engine->Prob(qubitIndex));
        }

        return engine->Prob(qubitIndex);
    }
    real1_f ProbAll(bitCapInt fullRegister)
    {
        FlushIfPhaseBlocked();
        return engine->ProbAll(fullRegister);
    }
    real1_f ProbMask(bitCapInt mask, bitCapInt permutation)
    {
        FlushBuffers();
        return engine->ProbMask(mask, permutation);
    }
    real1_f ProbParity(bitCapInt mask)
    {
        FlushBuffers();
        return engine->ProbParity(mask);
    }
    bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
    {
        FlushBuffers();
        return engine->ForceMParity(mask, result, doForce);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QMaskFusion>(toCompare));
    }
    real1_f SumSqrDiff(QMaskFusionPtr toCompare)
    {
        FlushBuffers();
        toCompare->FlushBuffers();
        return engine->SumSqrDiff(toCompare->engine);
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) { engine->UpdateRunningNorm(norm_thresh); }
    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        engine->NormalizeState(nrm, norm_thresh, phaseArg);
    }

    real1_f ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset = 0)
    {
        FlushIfPhaseBlocked(bits, length);
        return engine->ExpectationBitsAll(bits, length, offset);
    }

    bool TrySeparate(bitLenInt qubit) { return engine->TrySeparate(qubit); }
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2) { return engine->TrySeparate(qubit1, qubit2); }
    bool TrySeparate(const bitLenInt* qubits, bitLenInt length, real1_f error_tol)
    {
        return engine->TrySeparate(qubits, length, error_tol);
    }

    void Finish() { engine->Finish(); }

    bool isFinished() { return engine->isFinished(); }

    QInterfacePtr Clone();

    void SetDevice(int64_t dID)
    {
        devID = dID;
        engine->SetDevice(dID);
    }

    int64_t GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize() { return engine->GetMaxSize(); };

protected:
    real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
    {
        FlushBuffers();
        return engine->GetExpectation(valueStart, valueLength);
    }

    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        engine->Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }
    void ApplyControlled2x2(const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, const complex* mtrx)
    {
        engine->ApplyControlled2x2(controls, controlLen, target, mtrx);
    }
    void ApplyAntiControlled2x2(const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, const complex* mtrx)
    {
        engine->ApplyAntiControlled2x2(controls, controlLen, target, mtrx);
    }

    void FreeStateVec(complex* sv = NULL) { engine->FreeStateVec(sv); }

#if ENABLE_ALU
    void INCDECC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryIndex);
        engine->INCDECC(toMod, inOutStart, length, carryIndex);
    }
    void INCDECSC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryIndex);
        engine->INCDECSC(toMod, inOutStart, length, carryIndex);
    }
    void INCDECSC(
        bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(overflowIndex) || FlushIfBuffered(carryIndex);
        engine->INCDECSC(toMod, inOutStart, length, overflowIndex, carryIndex);
    }
#if ENABLE_BCD
    void INCDECBCDC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryIndex);
        engine->INCDECBCDC(toMod, inOutStart, length, carryIndex);
    }
#endif
#endif
};
} // namespace Qrack
