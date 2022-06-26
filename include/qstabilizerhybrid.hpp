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

#include "mpsshard.hpp"
#include "qengine.hpp"
#include "qstabilizer.hpp"

#define QINTERFACE_TO_QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QINTERFACE_TO_QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

namespace Qrack {

class QStabilizerHybrid;
typedef std::shared_ptr<QStabilizerHybrid> QStabilizerHybridPtr;

/**
 * A "Qrack::QStabilizerHybrid" internally switched between Qrack::QStabilizer and Qrack::QEngine to maximize
 * performance.
 */
class QStabilizerHybrid : public QEngine {
protected:
    bool isDefaultPaging;
    bool doNormalize;
    bool isSparse;
    bool useTGadget;
    bitLenInt thresholdQubits;
    bitLenInt maxPageQubits;
    bitLenInt ancillaCount;
    bitLenInt maxQubitPlusAncillaCount;
    real1_f separabilityThreshold;
    int64_t devID;
    complex phaseFactor;
    QEnginePtr engine;
    QStabilizerPtr stabilizer;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engineTypes;
    std::vector<MpsShardPtr> shards;

    QStabilizerPtr MakeStabilizer(bitCapInt perm = 0U);
    QEnginePtr MakeEngine(bitCapInt perm = 0U);
    QEnginePtr MakeEngine(bitCapInt perm, bitLenInt qbCount);

    void InvertBuffer(bitLenInt qubit);
    void FlushH(bitLenInt qubit);
    void FlushIfBlocked(bitLenInt control, bitLenInt target, bool isPhase = false);
    bool CollapseSeparableShard(bitLenInt qubit);
    bool TrimControls(
        const bitLenInt* lControls, bitLenInt lControlLen, std::vector<bitLenInt>& output, bool anti = false);
    void CacheEigenstate(bitLenInt target);
    void FlushBuffers();
    void DumpBuffers()
    {
        for (bitLenInt i = 0; i < shards.size(); ++i) {
            shards[i] = NULL;
        }
    }
    bool IsBuffered()
    {
        for (bitLenInt i = 0U; i < shards.size(); ++i) {
            if (shards[i]) {
                // We have a cached non-Clifford operation.
                return true;
            }
        }

        return false;
    }
    bool IsProbBuffered()
    {
        for (bitLenInt i = 0U; i < shards.size(); ++i) {
            MpsShardPtr shard = shards[i];
            if (shard && !((norm(shard->gate[1]) <= FP_NORM_EPSILON) && (norm(shard->gate[2]) <= FP_NORM_EPSILON))) {
                // We have a cached non-Clifford operation.
                return true;
            }
        }

        return false;
    }

    real1_f ApproxCompareHelper(
        QStabilizerHybridPtr toCompare, bool isDiscreteBool, real1_f error_tol = TRYDECOMPOSE_EPSILON);

public:
    QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0U,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QStabilizerHybrid(bitLenInt qBitCount, bitCapInt initState = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QStabilizerHybrid({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    void SetTInjection(bool useGadget) { useTGadget = useGadget; }

    void Finish()
    {
        if (stabilizer) {
            stabilizer->Finish();
        } else {
            engine->Finish();
        }
    };

    bool isFinished() { return (!stabilizer || stabilizer->isFinished()) && (!engine || engine->isFinished()); }

    void Dump()
    {
        if (stabilizer) {
            stabilizer->Dump();
        } else {
            engine->Dump();
        }
    }

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        if (engine) {
            SetConcurrency(GetConcurrencyLevel());
        }
    }

    void ZeroAmplitudes()
    {
        SwitchToEngine();
        engine->ZeroAmplitudes();
    }
    void CopyStateVec(QEnginePtr src) { CopyStateVec(std::dynamic_pointer_cast<QStabilizerHybrid>(src)); }
    void CopyStateVec(QStabilizerHybridPtr src)
    {
        SetPermutation(0U);

        if (src->stabilizer) {
            stabilizer = std::dynamic_pointer_cast<QStabilizer>(src->stabilizer->Clone());
            return;
        }

        engine = MakeEngine();
        engine->CopyStateVec(src->engine);
    }
    bool IsZeroAmplitude()
    {
        if (stabilizer) {
            return false;
        }

        return engine->IsZeroAmplitude();
    }
    void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        SwitchToEngine();
        engine->GetAmplitudePage(pagePtr, offset, length);
    }
    void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
    {
        SwitchToEngine();
        engine->SetAmplitudePage(pagePtr, offset, length);
    }
    void SetAmplitudePage(QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        SetAmplitudePage(std::dynamic_pointer_cast<QStabilizerHybrid>(pageEnginePtr), srcOffset, dstOffset, length);
    }
    void SetAmplitudePage(
        QStabilizerHybridPtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
    {
        SwitchToEngine();
        pageEnginePtr->SwitchToEngine();
        engine->SetAmplitudePage(pageEnginePtr->engine, srcOffset, dstOffset, length);
    }
    void ShuffleBuffers(QEnginePtr oEngine) { ShuffleBuffers(std::dynamic_pointer_cast<QStabilizerHybrid>(oEngine)); }
    void ShuffleBuffers(QStabilizerHybridPtr oEngine)
    {
        SwitchToEngine();
        oEngine->SwitchToEngine();
        engine->ShuffleBuffers(oEngine->engine);
    }
    QEnginePtr CloneEmpty();
    void QueueSetDoNormalize(bool doNorm)
    {
        doNormalize = doNorm;
        if (engine) {
            engine->QueueSetDoNormalize(doNorm);
        }
    }
    void QueueSetRunningNorm(real1_f runningNrm)
    {
        if (engine) {
            engine->QueueSetRunningNorm(runningNrm);
        }
    }
    real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
    {
        QStabilizerHybridPtr thisClone = stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;
        if (thisClone) {
            thisClone->SwitchToEngine();
        }
        QInterfacePtr thisEngine = thisClone ? thisClone->engine : engine;
        return thisEngine->ProbReg(start, length, permutation);
    }
    using QEngine::ApplyM;
    void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
    {
        SwitchToEngine();
        return engine->ApplyM(regMask, result, nrm);
    }
    real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
    {
        QStabilizerHybridPtr thisClone = stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;
        if (thisClone) {
            thisClone->SwitchToEngine();
        }
        QEnginePtr thisEngine = thisClone ? thisClone->engine : engine;
        return thisEngine->GetExpectation(valueStart, valueLength);
    }
    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        SwitchToEngine();
        engine->Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }
    void FreeStateVec(complex* sv = NULL)
    {
        SwitchToEngine();
        engine->FreeStateVec(sv);
    }
    real1_f GetRunningNorm()
    {
        if (stabilizer) {
            return (real1_f)ONE_R1;
        }

        Finish();
        return engine->GetRunningNorm();
    }

    real1_f FirstNonzeroPhase()
    {
        if (stabilizer) {
            return stabilizer->FirstNonzeroPhase();
        }

        return engine->FirstNonzeroPhase();
    }

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    void SwitchToEngine();

    bool isClifford() { return !engine; }

    bool isClifford(bitLenInt qubit) { return !engine && !shards[qubit]; };

    bool isBinaryDecisionTree() { return engine && engine->isBinaryDecisionTree(); };

    using QEngine::Compose;
    bitLenInt Compose(QStabilizerHybridPtr toCopy);
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy)); }
    bitLenInt Compose(QStabilizerHybridPtr toCopy, bitLenInt start);
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy), start);
    }
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QStabilizerHybrid>(dest));
    }
    void Decompose(bitLenInt start, QStabilizerHybridPtr dest);
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    void GetQuantumState(complex* outputState);
    void GetProbs(real1* outputProbs);
    complex GetAmplitude(bitCapInt perm);
    void SetQuantumState(const complex* inputState);
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        SwitchToEngine();
        engine->SetAmplitude(perm, amp);
    }
    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        DumpBuffers();

        engine = NULL;

        if (stabilizer) {
            stabilizer->SetPermutation(perm);
        } else {
            stabilizer = MakeStabilizer(perm);
        }
    }

    void Swap(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit1 == qubit2) {
            return;
        }

        std::swap(shards[qubit1], shards[qubit2]);

        if (stabilizer) {
            stabilizer->Swap(qubit1, qubit2);
        } else {
            engine->Swap(qubit1, qubit2);
        }
    }

    void ISwap(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit1 == qubit2) {
            return;
        }

        MpsShardPtr shard = shards[qubit1];
        if (shard && (shard->IsHPhase() || shard->IsHInvert())) {
            FlushH(qubit1);
        }
        shard = shards[qubit1];
        if (shard && shard->IsInvert()) {
            InvertBuffer(qubit1);
        }

        shard = shards[qubit2];
        if (shard && (shard->IsHPhase() || shard->IsHInvert())) {
            FlushH(qubit2);
        }
        shard = shards[qubit2];
        if (shard && shard->IsInvert()) {
            InvertBuffer(qubit2);
        }

        if ((shards[qubit1] && !shards[qubit1]->IsPhase()) || (shards[qubit2] && !shards[qubit2]->IsPhase())) {
            FlushBuffers();
        }

        if (stabilizer) {
            stabilizer->ISwap(qubit1, qubit2);
        } else {
            engine->ISwap(qubit1, qubit2);
        }
    }

    real1_f Prob(bitLenInt qubit);

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    bitCapInt MAll();

    void Mtrx(const complex* mtrx, bitLenInt target);
    void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    void MCInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);
    void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    void MACPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    void MACInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);

    using QEngine::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(
        const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex, const complex* mtrxs)
    {
        if (stabilizer) {
            QInterface::UniformlyControlledSingleBit(controls, controlLen, qubitIndex, mtrxs);
            return;
        }

        engine->UniformlyControlledSingleBit(controls, controlLen, qubitIndex, mtrxs);
    }

    void CSwap(const bitLenInt* lControls, bitLenInt lControlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, lControlLen, controls, false)) {
                return;
            }
            if (!controls.size()) {
                stabilizer->Swap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->CSwap(lControls, lControlLen, qubit1, qubit2);
    }
    void CSqrtSwap(const bitLenInt* lControls, bitLenInt lControlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, lControlLen, controls, false)) {
                return;
            }
            if (!controls.size()) {
                QInterface::SqrtSwap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->CSqrtSwap(lControls, lControlLen, qubit1, qubit2);
    }
    void AntiCSqrtSwap(const bitLenInt* lControls, bitLenInt lControlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, lControlLen, controls, true)) {
                return;
            }
            if (!controls.size()) {
                QInterface::SqrtSwap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->AntiCSqrtSwap(lControls, lControlLen, qubit1, qubit2);
    }
    void CISqrtSwap(const bitLenInt* lControls, bitLenInt lControlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, lControlLen, controls, false)) {
                return;
            }
            if (!controls.size()) {
                QInterface::ISqrtSwap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->CISqrtSwap(lControls, lControlLen, qubit1, qubit2);
    }
    void AntiCISqrtSwap(const bitLenInt* lControls, bitLenInt lControlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        if (stabilizer) {
            std::vector<bitLenInt> controls;
            if (TrimControls(lControls, lControlLen, controls, true)) {
                return;
            }
            if (!controls.size()) {
                QInterface::ISqrtSwap(qubit1, qubit2);
                return;
            }
            SwitchToEngine();
        }

        engine->AntiCISqrtSwap(lControls, lControlLen, qubit1, qubit2);
    }

    void XMask(bitCapInt mask)
    {
        if (!stabilizer) {
            engine->XMask(mask);
            return;
        }

        bitCapInt v = mask;
        while (mask) {
            v = v & (v - ONE_BCI);
            X(log2(mask ^ v));
            mask = v;
        }
    }

    void YMask(bitCapInt mask)
    {
        if (!stabilizer) {
            engine->YMask(mask);
            return;
        }

        bitCapInt v = mask;
        while (mask) {
            v = v & (v - ONE_BCI);
            Y(log2(mask ^ v));
            mask = v;
        }
    }

    void ZMask(bitCapInt mask)
    {
        if (!stabilizer) {
            engine->ZMask(mask);
            return;
        }

        bitCapInt v = mask;
        while (mask) {
            v = v & (v - ONE_BCI);
            Z(log2(mask ^ v));
            mask = v;
        }
    }

    std::map<bitCapInt, int> MultiShotMeasureMask(const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots);
    void MultiShotMeasureMask(
        const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots, unsigned long long* shotsArray);

    real1_f ProbParity(bitCapInt mask)
    {
        if (!mask) {
            return ZERO_R1_F;
        }

        if (!(mask & (mask - ONE_BCI))) {
            return Prob(log2(mask));
        }

        SwitchToEngine();
        return QINTERFACE_TO_QPARITY(engine)->ProbParity(mask);
    }
    bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
    {
        // If no bits in mask:
        if (!mask) {
            return false;
        }

        // If only one bit in mask:
        if (!(mask & (mask - ONE_BCI))) {
            return ForceM(log2(mask), result, doForce);
        }

        SwitchToEngine();
        return QINTERFACE_TO_QPARITY(engine)->ForceMParity(mask, result, doForce);
    }
    void UniformParityRZ(bitCapInt mask, real1_f angle)
    {
        SwitchToEngine();
        QINTERFACE_TO_QPARITY(engine)->UniformParityRZ(mask, angle);
    }
    void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
    {
        SwitchToEngine();
        QINTERFACE_TO_QPARITY(engine)->CUniformParityRZ(controls, controlLen, mask, angle);
    }

#if ENABLE_ALU
    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->PhaseFlipIfLess(greaterPerm, start, length);
    }

    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        if (stabilizer) {
            QInterface::INC(toAdd, start, length);
            return;
        }

        engine->INC(toAdd, start, length);
    }
    void CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        if (stabilizer) {
            QInterface::CINC(toAdd, inOutStart, length, controls, controlLen);
            return;
        }

        engine->CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (stabilizer) {
            QInterface::INCS(toAdd, start, length, overflowIndex);
            return;
        }

        engine->INCS(toAdd, start, length, overflowIndex);
    }
    void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        if (stabilizer) {
            QInterface::INCDECC(toAdd, start, length, carryIndex);
            return;
        }

        engine->INCDECC(toAdd, start, length, carryIndex);
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, carryIndex);
    }
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCBCD(toAdd, start, length);
    }
    void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECBCDC(toAdd, start, length, carryIndex);
    }
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MUL(toMul, inOutStart, carryStart, length);
    }
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->DIV(toDiv, inOutStart, carryStart, length);
    }
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedLDA(
            indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedADC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedSBC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->Hash(start, length, values);
    }
#endif

    void PhaseFlip()
    {
        if (stabilizer) {
            stabilizer->PhaseFlip();
        } else {
            engine->PhaseFlip();
        }
    }
    void ZeroPhaseFlip(bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->ZeroPhaseFlip(start, length);
    }

    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            QInterface::SqrtSwap(qubitIndex1, qubitIndex2);
            return;
        }

        SwitchToEngine();
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
    }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            QInterface::ISqrtSwap(qubitIndex1, qubitIndex2);
            return;
        }

        SwitchToEngine();
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    real1_f ProbMask(bitCapInt mask, bitCapInt permutation)
    {
        SwitchToEngine();
        return engine->ProbMask(mask, permutation);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), false);
    }
    bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return error_tol >=
            ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), true, error_tol);
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->UpdateRunningNorm(norm_thresh);
        }
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);

    real1_f ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset = 0)
    {
        if (stabilizer) {
            return QInterface::ExpectationBitsAll(bits, length, offset);
        }

        return engine->ExpectationBitsAll(bits, length, offset);
    }

    bool TrySeparate(bitLenInt qubit);
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2);
    bool TrySeparate(const bitLenInt* qubits, bitLenInt length, real1_f error_tol);

    QInterfacePtr Clone();

    void SetDevice(int64_t dID)
    {
        devID = dID;
        if (engine) {
            engine->SetDevice(dID);
        }
    }

    int64_t GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize()
    {
        if (stabilizer) {
            return QInterface::GetMaxSize();
        }

        return engine->GetMaxSize();
    }
};
} // namespace Qrack
