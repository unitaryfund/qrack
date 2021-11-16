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

#include "mpsshard.hpp"
#include "qengine.hpp"
#include "qstabilizer.hpp"

#include "common/qrack_types.hpp"

#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif

namespace Qrack {

class QStabilizerHybrid;
typedef std::shared_ptr<QStabilizerHybrid> QStabilizerHybridPtr;

/**
 * A "Qrack::QStabilizerHybrid" internally switched between Qrack::QEngineCPU and Qrack::QEngineOCL to maximize
 * qubit-count-dependent performance.
 */
class QStabilizerHybrid : public QInterface {
protected:
    std::vector<QInterfaceEngine> engineTypes;
    QInterfacePtr engine;
    QStabilizerPtr stabilizer;
    std::vector<MpsShardPtr> shards;
    int devID;
    complex phaseFactor;
    bool doNormalize;
    bool useHostRam;
    bool useRDRAND;
    bool isSparse;
    real1_f separabilityThreshold;
    uint32_t concurrency;
    bitLenInt thresholdQubits;

    QStabilizerPtr MakeStabilizer(const bitCapInt& perm = 0);
    QInterfacePtr MakeEngine(const bitCapInt& perm = 0);

    void InvertBuffer(bitLenInt qubit)
    {
        complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
        MpsShardPtr pauliShard = std::make_shared<MpsShard>(pauliX);
        pauliShard->Compose(shards[qubit]->gate);
        shards[qubit] = pauliShard->IsIdentity() ? NULL : pauliShard;
        stabilizer->X(qubit);
    }

    void FlushIfBlocked(bitLenInt control, bitLenInt target, bool isPhase = false)
    {
        if (engine) {
            return;
        }

        if (shards[target] && shards[target]->IsInvert()) {
            InvertBuffer(target);
        }

        if (shards[control] && shards[control]->IsInvert()) {
            InvertBuffer(control);
        }

        bool isBlocked = (shards[target] && (!isPhase || !shards[target]->IsPhase()));
        isBlocked |= (shards[control] && !shards[control]->IsPhase());

        if (isBlocked) {
            SwitchToEngine();
        }
    }

    virtual bool CollapseSeparableShard(bitLenInt qubit)
    {
        MpsShardPtr shard = shards[qubit];
        shards[qubit] = NULL;
        real1_f prob;

        bool isZ1 = stabilizer->M(qubit);

        if (isZ1) {
            prob = norm(shard->gate[3]);
        } else {
            prob = norm(shard->gate[2]);
        }

        bool result;
        if (prob <= ZERO_R1) {
            result = false;
        } else if (prob >= ONE_R1) {
            result = true;
        } else {
            result = (Rand() <= prob);
        }

        if (result != isZ1) {
            stabilizer->X(qubit);
        }

        return result;
    }

    virtual void FlushBuffers()
    {
        bitLenInt i;

        if (stabilizer) {
            for (i = 0; i < qubitCount; i++) {
                if (shards[i]) {
                    // This will call FlushBuffers() again after no longer stabilizer.
                    SwitchToEngine();
                    return;
                }
            }
        }

        if (stabilizer) {
            return;
        }

        for (i = 0; i < qubitCount; i++) {
            MpsShardPtr shard = shards[i];
            if (shard) {
                shards[i] = NULL;
                ApplySingleBit(shard->gate, i);
            }
        }
    }

    virtual void DumpBuffers()
    {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            shards[i] = NULL;
        }
    }

    virtual bool TrimControls(const bitLenInt* lControls, const bitLenInt& lControlLen, std::vector<bitLenInt>& output,
        const bool& anti = false)
    {
        if (engine) {
            output.insert(output.begin(), lControls, lControls + lControlLen);
            return false;
        }

        bitLenInt bit;
        for (bitLenInt i = 0; i < lControlLen; i++) {
            bit = lControls[i];

            if (!stabilizer->IsSeparableZ(bit)) {
                output.push_back(bit);
                continue;
            }

            if (shards[bit]) {
                if (shards[bit]->IsInvert()) {
                    InvertBuffer(bit);

                    if (!shards[bit]) {
                        if (anti == stabilizer->M(bit)) {
                            return true;
                        }
                        continue;
                    }
                }

                if (shards[bit]->IsPhase()) {
                    if (anti == stabilizer->M(bit)) {
                        return true;
                    }
                    continue;
                }

                output.push_back(bit);
            } else if (anti == stabilizer->M(bit)) {
                return true;
            }
        }

        return false;
    }

    virtual void CacheEigenstate(const bitLenInt& target);

public:
    QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QStabilizerHybrid(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
#if ENABLE_OPENCL
        : QStabilizerHybrid(
              { OCLEngine::Instance()->GetDeviceCount() ? QINTERFACE_QPAGER : QINTERFACE_OPTIMAL_G2_CHILD }, qBitCount,
              initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId, useHardwareRNG,
              useSparseStateVec, norm_thresh, ignored, qubitThreshold, separation_thresh)
#else
        : QStabilizerHybrid({ QINTERFACE_OPTIMAL_G1_CHILD }, qBitCount, initState, rgp, phaseFac, doNorm,
              randomGlobalPhase, useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, ignored,
              qubitThreshold, separation_thresh)
#endif
    {
    }

    virtual void Finish()
    {
        if (stabilizer) {
            stabilizer->Finish();
        } else {
            engine->Finish();
        }
    };

    virtual bool isFinished() { return (!stabilizer || stabilizer->isFinished()) && (!engine || engine->isFinished()); }

    virtual void SetConcurrency(uint32_t threadCount)
    {
        concurrency = threadCount;
        if (engine) {
            SetConcurrency(concurrency);
        }
    }

    virtual void TurnOnPaging()
    {
        if (engineTypes[0] == QINTERFACE_QPAGER) {
            return;
        }
        engineTypes.insert(engineTypes.begin(), QINTERFACE_QPAGER);

        if (engine) {
            QPagerPtr nEngine = std::dynamic_pointer_cast<QPager>(MakeEngine());
            nEngine->LockEngine(std::dynamic_pointer_cast<QEngine>(engine));
            engine = nEngine;
        }
    }

    virtual void TurnOffPaging()
    {
        if (engineTypes[0] != QINTERFACE_QPAGER) {
            return;
        }
        engineTypes.erase(engineTypes.begin());
        if (!engineTypes.size()) {
            engineTypes.push_back(QINTERFACE_OPTIMAL_SINGLE_PAGE);
        }

        if (engine) {
            engine = std::dynamic_pointer_cast<QPager>(engine)->ReleaseEngine();
        }
    }

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    virtual void SwitchToEngine();

    virtual bool isClifford() { return !engine; }

    virtual bool isClifford(const bitLenInt& qubit) { return !engine && !(shards[qubit]); };

    virtual bool isBinaryDecisionTree() { return engine && engine->isBinaryDecisionTree(); };

    using QInterface::Compose;
    virtual bitLenInt Compose(QStabilizerHybridPtr toCopy)
    {
        bitLenInt toRet;

        if (engine) {
            toCopy->SwitchToEngine();
            toRet = engine->Compose(toCopy->engine);
        } else if (toCopy->engine) {
            SwitchToEngine();
            toRet = engine->Compose(toCopy->engine);
        } else {
            toRet = stabilizer->Compose(toCopy->stabilizer);
        }

        shards.insert(shards.end(), toCopy->shards.begin(), toCopy->shards.end());

        SetQubitCount(qubitCount + toCopy->qubitCount);

        return toRet;
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy));
    }
    virtual bitLenInt Compose(QStabilizerHybridPtr toCopy, bitLenInt start)
    {
        bitLenInt toRet;

        if (engine) {
            toCopy->SwitchToEngine();
            toRet = engine->Compose(toCopy->engine, start);
        } else if (toCopy->engine) {
            SwitchToEngine();
            toRet = engine->Compose(toCopy->engine, start);
        } else {
            toRet = stabilizer->Compose(toCopy->stabilizer, start);
        }

        shards.insert(shards.begin() + start, toCopy->shards.begin(), toCopy->shards.end());

        SetQubitCount(qubitCount + toCopy->qubitCount);

        return toRet;
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QStabilizerHybrid>(dest));
    }
    virtual void Decompose(bitLenInt start, QStabilizerHybridPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState)
    {
        FlushBuffers();

        if (stabilizer) {
            stabilizer->GetQuantumState(outputState);
        } else {
            engine->GetQuantumState(outputState);
        }
    }
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm)
    {
        SwitchToEngine();
        return engine->GetAmplitude(perm);
    }
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        SwitchToEngine();
        engine->SetAmplitude(perm, amp);
    }
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        DumpBuffers();

        engine = NULL;

        if (stabilizer) {
            stabilizer->SetPermutation(perm);
        } else {
            stabilizer = MakeStabilizer(perm);
        }
    }

    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2)
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

    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit1 == qubit2) {
            return;
        }

        if (shards[qubit1] || shards[qubit2]) {
            FlushBuffers();
        }

        if (stabilizer) {
            stabilizer->ISwap(qubit1, qubit2);
        } else {
            engine->ISwap(qubit1, qubit2);
        }
    }

    virtual real1_f Prob(bitLenInt qubit);

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    virtual bitCapInt MAll();

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt target);

    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target);

    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target);

    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);

    virtual void ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);

    virtual void ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);

    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);

    virtual void ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);

    virtual void ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);

    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask)
    {
        // If there are no controls, this is equivalent to the single bit gate.
        if (!controlLen) {
            ApplySingleBit(mtrxs, qubitIndex);
            return;
        }

        SwitchToEngine();
        engine->UniformlyControlledSingleBit(
            controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
    }

    virtual void UniformParityRZ(const bitCapInt& mask, const real1_f& angle)
    {
        SwitchToEngine();
        engine->UniformParityRZ(mask, angle);
    }

    virtual void CUniformParityRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle)
    {
        SwitchToEngine();
        engine->CUniformParityRZ(controls, controlLen, mask, angle);
    }

    virtual void CSwap(
        const bitLenInt* lControls, const bitLenInt& lControlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        std::vector<bitLenInt> controls;
        if (TrimControls(lControls, lControlLen, controls)) {
            return;
        }

        if (!controls.size()) {
            Swap(qubit1, qubit2);
            return;
        }

        SwitchToEngine();
        engine->CSwap(lControls, lControlLen, qubit1, qubit2);
    }
    virtual void AntiCSwap(
        const bitLenInt* lControls, const bitLenInt& lControlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        std::vector<bitLenInt> controls;
        if (TrimControls(lControls, lControlLen, controls, true)) {
            return;
        }

        if (!controls.size()) {
            Swap(qubit1, qubit2);
            return;
        }

        SwitchToEngine();
        engine->AntiCSwap(lControls, lControlLen, qubit1, qubit2);
    }
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        SwitchToEngine();
        engine->CSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        SwitchToEngine();
        engine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        SwitchToEngine();
        engine->CISqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        SwitchToEngine();
        engine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
    }

    virtual void XMask(bitCapInt mask)
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

    virtual void YMask(bitCapInt mask)
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

    virtual void ZMask(bitCapInt mask)
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

    virtual std::map<bitCapInt, int> MultiShotMeasureMask(
        const bitCapInt* qPowers, const bitLenInt qPowerCount, const unsigned int shots);

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->INC(toAdd, start, length);
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        engine->CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->INCC(toAdd, start, length, carryIndex);
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        SwitchToEngine();
        engine->INCS(toAdd, start, length, overflowIndex);
    }
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->INCSC(toAdd, start, length, carryIndex);
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->DECC(toSub, start, length, carryIndex);
    }
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->DECSC(toSub, start, length, carryIndex);
    }
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->INCBCD(toAdd, start, length);
    }
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->INCBCDC(toAdd, start, length, carryIndex);
    }
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->DECBCDC(toSub, start, length, carryIndex);
    }
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        engine->MUL(toMul, inOutStart, carryStart, length);
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        engine->DIV(toDiv, inOutStart, carryStart, length);
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        engine->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        engine->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        engine->POWModNOut(base, modN, inStart, outStart, length);
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        engine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        engine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        SwitchToEngine();
        engine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
    }

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->ZeroPhaseFlip(start, length);
    }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        SwitchToEngine();
        engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->PhaseFlipIfLess(greaterPerm, start, length);
    }
    virtual void PhaseFlip()
    {
        if (engine) {
            engine->PhaseFlip();
        }
    }

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true)
    {
        SwitchToEngine();
        return engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        SwitchToEngine();
        return engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        SwitchToEngine();
        return engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values)
    {
        SwitchToEngine();
        engine->Hash(start, length, values);
    }

    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
    }
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    virtual real1_f ProbAll(bitCapInt fullRegister)
    {
        SwitchToEngine();
        return engine->ProbAll(fullRegister);
    }
    virtual real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        SwitchToEngine();
        return engine->ProbMask(mask, permutation);
    }
    // TODO: Good opportunity to optimize
    virtual real1_f ProbParity(const bitCapInt& mask)
    {
        if (!mask) {
            return ZERO_R1;
        }

        if (!(mask & (mask - ONE_BCI))) {
            return Prob(log2(mask));
        }

        SwitchToEngine();
        return engine->ProbParity(mask);
    }
    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true)
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
        return engine->ForceMParity(mask, result, doForce);
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare));
    }

    virtual real1_f SumSqrDiff(QStabilizerHybridPtr toCompare)
    {
        // If the qubit counts are unequal, these can't be approximately equal objects.
        if (qubitCount != toCompare->qubitCount) {
            // Max square difference:
            return ONE_R1;
        }

        QStabilizerHybridPtr thisClone = stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;
        QStabilizerHybridPtr thatClone =
            toCompare->stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;

        if (thisClone) {
            thisClone->SwitchToEngine();
        }
        if (thatClone) {
            thatClone->SwitchToEngine();
        }

        QInterfacePtr thisEngine = thisClone ? thisClone->engine : engine;
        QInterfacePtr thatEngine = thatClone ? thatClone->engine : toCompare->engine;

        return thisEngine->SumSqrDiff(thatEngine);
    }

    virtual bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), error_tol);
    }

    virtual bool ApproxCompare(QStabilizerHybridPtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        FlushBuffers();
        toCompare->FlushBuffers();

        if (stabilizer && toCompare->stabilizer) {
            return stabilizer->ApproxCompare(toCompare->stabilizer);
        }

        QStabilizerHybridPtr thisClone = stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;
        QStabilizerHybridPtr thatClone =
            toCompare->stabilizer ? std::dynamic_pointer_cast<QStabilizerHybrid>(Clone()) : NULL;

        if (thisClone) {
            thisClone->SwitchToEngine();
        }
        if (thatClone) {
            thatClone->SwitchToEngine();
        }

        QInterfacePtr thisEngine = thisClone ? thisClone->engine : engine;
        QInterfacePtr thatEngine = thatClone ? thatClone->engine : toCompare->engine;

        return thisEngine->ApproxCompare(thatEngine, error_tol);
    }

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->UpdateRunningNorm(norm_thresh);
        }
    }

    virtual void NormalizeState(real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->NormalizeState(nrm, norm_thresh);
        }
    }

    virtual real1_f ExpectationBitsAll(const bitLenInt* bits, const bitLenInt& length, const bitCapInt& offset = 0)
    {
        if (stabilizer) {
            return QInterface::ExpectationBitsAll(bits, length, offset);
        }

        return engine->ExpectationBitsAll(bits, length, offset);
    }

    virtual bool TrySeparate(bitLenInt qubit)
    {
        if (qubitCount == 1U) {
            return true;
        }

        if (stabilizer) {
            return stabilizer->CanDecomposeDispose(qubit, 1);
        }

        return engine->TrySeparate(qubit);
    }
    virtual bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubitCount == 2U) {
            return true;
        }

        if (stabilizer) {
            if (qubit2 < qubit1) {
                std::swap(qubit1, qubit2);
            }

            stabilizer->Swap(qubit1 + 1U, qubit2);

            bool toRet = stabilizer->CanDecomposeDispose(qubit1, 2);

            stabilizer->Swap(qubit1 + 1U, qubit2);

            return toRet;
        }

        return engine->TrySeparate(qubit1, qubit2);
    }
    virtual bool TrySeparate(bitLenInt* qubits, bitLenInt length, real1_f error_tol)
    {
        if (stabilizer) {
            std::vector<bitLenInt> q(length);
            std::copy(qubits, qubits + length, q.begin());
            std::sort(q.begin(), q.end());

            for (bitLenInt i = 1; i < length; i++) {
                Swap(q[0] + i, q[i]);
            }

            bool toRet = stabilizer->CanDecomposeDispose(q[0], length);

            for (bitLenInt i = 1; i < length; i++) {
                Swap(q[0] + i, q[i]);
            }

            return toRet;
        }

        return engine->TrySeparate(qubits, length, error_tol);
    }

    virtual QInterfacePtr Clone();

    virtual void SetDevice(const int& dID, const bool& forceReInit = false)
    {
        devID = dID;
        if (engine) {
            engine->SetDevice(dID, forceReInit);
        }
    }

    virtual int64_t GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize()
    {
        if (stabilizer) {
            return QInterface::GetMaxSize();
        }

        return engine->GetMaxSize();
    }
};
} // namespace Qrack
