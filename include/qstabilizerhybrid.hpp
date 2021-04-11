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
#include "qstabilizer.hpp"

#if ENABLE_QUNIT_CPU_PARALLEL
#include "common/dispatchqueue.hpp"
#endif

namespace Qrack {

struct QStabilizerShard;
typedef std::shared_ptr<QStabilizerShard> QStabilizerShardPtr;

class QStabilizerHybrid;
typedef std::shared_ptr<QStabilizerHybrid> QStabilizerHybridPtr;

struct QStabilizerShard {
    complex gate[4];

    QStabilizerShard()
    {
        gate[0] = ONE_CMPLX;
        gate[1] = ZERO_CMPLX;
        gate[2] = ZERO_CMPLX;
        gate[3] = ONE_CMPLX;
    }

    QStabilizerShard(complex* g) { std::copy(g, g + 4, gate); }

    void Compose(const complex* g)
    {
        complex o[4];
        std::copy(gate, gate + 4, o);
        mul2x2((complex*)g, o, gate);
    }

    bool IsPhase() { return (norm(gate[1]) <= FP_NORM_EPSILON) && (norm(gate[2]) <= FP_NORM_EPSILON); }

    bool IsInvert() { return (norm(gate[0]) <= FP_NORM_EPSILON) && (norm(gate[3]) <= FP_NORM_EPSILON); }
};

/**
 * A "Qrack::QStabilizerHybrid" internally switched between Qrack::QEngineCPU and Qrack::QEngineOCL to maximize
 * qubit-count-dependent performance.
 */
class QStabilizerHybrid : public QInterface {
protected:
    QInterfaceEngine engineType;
    QInterfaceEngine subEngineType;
    QInterfacePtr engine;
    QStabilizerPtr stabilizer;
    std::vector<QStabilizerShardPtr> shards;
    int devID;
    complex phaseFactor;
    bool doNormalize;
    bool useHostRam;
    bool useRDRAND;
    bool isSparse;
    uint32_t concurrency;
    bitLenInt thresholdQubits;

    QStabilizerPtr MakeStabilizer(const bitCapInt& perm = 0);
    QInterfacePtr MakeEngine(const bitCapInt& perm = 0);

public:
    QStabilizerHybrid(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0);

    QStabilizerHybrid(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0)
        : QStabilizerHybrid(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem,
              deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold)
    {
    }

    QStabilizerHybrid(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0)
        : QStabilizerHybrid(QINTERFACE_OPTIMAL_SCHROEDINGER, QINTERFACE_OPTIMAL_SINGLE_PAGE, qBitCount, initState, rgp,
              phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh,
              ignored, qubitThreshold)
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
            QStabilizerShardPtr shard = shards[i];
            if (shard) {
                shards[i] = NULL;
                ApplySingleBit(shard->gate, i);
            }
        }
    }

    virtual void DumpBuffers()
    {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            if (shards[i]) {
                shards[i] = NULL;
            }
        }
    }

    virtual bool TrimControls(const bitLenInt* lControls, const bitLenInt& lControlLen, std::vector<bitLenInt>& output,
        const bool& anti = false)
    {
        if (engine) {
            output.insert(output.begin(), lControls, lControls + lControlLen);
            return false;
        }

        real1_f prob;
        for (bitLenInt i = 0; i < lControlLen; i++) {
            prob = Prob(lControls[i]);
            if (anti) {
                prob = ONE_R1 - prob;
            }

            if (prob == ZERO_R1) {
                return true;
            }
            if (prob != ONE_R1) {
                output.push_back(lControls[i]);
            }
        }

        return false;
    }

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    virtual void SwitchToEngine();

    virtual bool isClifford() { return !engine; }

    virtual bool isClifford(const bitLenInt& qubit) { return !engine && !(shards[qubit]); };

    /// Apply a CNOT gate with control and target
    virtual void CNOT(bitLenInt control, bitLenInt target)
    {
        FlushBuffers();

        if (stabilizer) {
            stabilizer->CNOT(control, target);
        } else {
            engine->CNOT(control, target);
        }
    }

    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);

    /// Apply a Hadamard gate to target
    virtual void H(bitLenInt target)
    {
        if (shards[target]) {
            complex mtrx[4] = { complex((real1)M_SQRT1_2, ZERO_R1), complex((real1)M_SQRT1_2, ZERO_R1),
                complex((real1)M_SQRT1_2, ZERO_R1), complex((real1)-M_SQRT1_2, ZERO_R1) };
            ApplySingleBit(mtrx, target);
            return;
        }

        if (stabilizer) {
            stabilizer->H(target);
        } else {
            engine->H(target);
        }
    }

    virtual void CH(bitLenInt control, bitLenInt target);

    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    virtual void S(bitLenInt target)
    {
        if (shards[target]) {
            complex mtrx[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
            ApplySingleBit(mtrx, target);
            return;
        }

        if (stabilizer) {
            stabilizer->S(target);
        } else {
            engine->S(target);
        }
    }

    virtual void CS(bitLenInt control, bitLenInt target);

    // TODO: Custom implementations for decompositions:
    virtual void Z(bitLenInt target)
    {
        if (shards[target]) {
            complex mtrx[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
            ApplySingleBit(mtrx, target);
            return;
        }

        if (stabilizer) {
            stabilizer->Z(target);
        } else {
            engine->Z(target);
        }
    }

    virtual void IS(bitLenInt target)
    {
        if (shards[target]) {
            complex mtrx[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };
            ApplySingleBit(mtrx, target);
            return;
        }

        if (stabilizer) {
            stabilizer->IS(target);
        } else {
            engine->IS(target);
        }
    }

    virtual void CIS(bitLenInt control, bitLenInt target);

    virtual void X(bitLenInt target)
    {
        if (shards[target]) {
            complex mtrx[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
            ApplySingleBit(mtrx, target);
            return;
        }

        if (stabilizer) {
            stabilizer->X(target);
        } else {
            engine->X(target);
        }
    }

    virtual void Y(bitLenInt target)
    {
        if (shards[target]) {
            complex mtrx[4] = { ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
            ApplySingleBit(mtrx, target);
            return;
        }

        if (stabilizer) {
            stabilizer->Y(target);
        } else {
            engine->Y(target);
        }
    }

    virtual void CZ(bitLenInt control, bitLenInt target)
    {
        FlushBuffers();

        if (stabilizer) {
            stabilizer->CZ(control, target);
        } else {
            engine->CZ(control, target);
        }
    }

    virtual void CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target);

    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2)
    {
        if (qubit1 == qubit2) {
            return;
        }

        FlushBuffers();

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

        FlushBuffers();

        if (stabilizer) {
            stabilizer->ISwap(qubit1, qubit2);
        } else {
            engine->ISwap(qubit1, qubit2);
        }
    }

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
        FlushBuffers();
        toCopy->FlushBuffers();

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

        if (stabilizer) {
            stabilizer->SetPermutation(perm);
        } else {
            engine->SetPermutation(perm, phaseFac);
        }
    }

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

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        QStabilizerShardPtr shard = shards[qubit];
        if (stabilizer && shard) {
            if (shard->IsInvert()) {
                X(qubit);
                shards[qubit] = NULL;
            } else if (shard->IsPhase()) {
                shards[qubit] = NULL;
            } else {
                FlushBuffers();
            }
        }

        // TODO: QStabilizer appears not to be decomposable after measurement and in many cases where a bit is in an
        // eigenstate.
        if (stabilizer &&
            (stabilizer->IsSeparableZ(qubit) ||
                ((engineType == QINTERFACE_QUNIT) || (engineType == QINTERFACE_QUNIT_MULTI)))) {
            return stabilizer->M(qubit, result, doForce, doApply);
        }

        SwitchToEngine();
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    virtual bitCapInt MAll();

    virtual std::map<bitCapInt, int> MultiShotMeasureMask(
        const bitCapInt* qPowers, const bitLenInt qPowerCount, const unsigned int shots)
    {
        SwitchToEngine();
        return engine->MultiShotMeasureMask(qPowers, qPowerCount, shots);
    }

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

    virtual real1_f Prob(bitLenInt qubitIndex)
    {
        bool isInvert = false;
        QStabilizerShardPtr shard = shards[qubitIndex];
        if (stabilizer && shard) {
            if (shard->IsInvert()) {
                isInvert = true;
            } else if (!shard->IsPhase()) {
                FlushBuffers();
            }
        }

        if (engine) {
            return engine->Prob(qubitIndex);
        }

        if (stabilizer->IsSeparableZ(qubitIndex)) {
            return stabilizer->M(qubitIndex) ^ isInvert ? ONE_R1 : ZERO_R1;
        } else {
            return ONE_R1 / 2;
        }
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

        if (qubitCount == 1U) {
            return Prob(0);
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
            return 4.0f;
        }

        SwitchToEngine();
        toCompare->SwitchToEngine();

        return engine->SumSqrDiff(toCompare->engine);
    }
    virtual bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = REAL1_EPSILON)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), error_tol);
    }

    virtual bool ApproxCompare(QStabilizerHybridPtr toCompare, real1_f error_tol = REAL1_EPSILON)
    {
        FlushBuffers();
        toCompare->FlushBuffers();

        if (!stabilizer == !(toCompare->engine)) {
            SwitchToEngine();
            toCompare->SwitchToEngine();
        }

        if (stabilizer) {
            return stabilizer->ApproxCompare(toCompare->stabilizer);
        }

        return engine->ApproxCompare(toCompare->engine, error_tol);
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

    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1, real1_f error_tol = REAL1_EPSILON)
    {
        if (stabilizer) {
            return stabilizer->CanDecomposeDispose(start, length);
        }

        return engine->TrySeparate(start, length, error_tol);
    }

    virtual QInterfacePtr Clone();

    virtual void SetDevice(const int& dID, const bool& forceReInit = false)
    {
        devID = dID;
        if (engine) {
            engine->SetDevice(dID, forceReInit);
        }
    }

    virtual int GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize()
    {
        if (stabilizer) {
            return QInterface::GetMaxSize();
        }

        return engine->GetMaxSize();
    }
};
} // namespace Qrack
