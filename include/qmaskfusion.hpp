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

struct QMaskFusionShard {
    bool isX;
    bool isZ;
    bool isBuffered() { return isX || isZ; }
};

class QMaskFusion;
typedef std::shared_ptr<QMaskFusion> QMaskFusionPtr;

/**
 * A "Qrack::QMaskFusion" internally switched between Qrack::QEngineCPU and Qrack::QEngineOCL to maximize
 * qubit-count-dependent performance.
 */
class QMaskFusion : public QInterface {
protected:
    QInterfacePtr engine;
    QInterfaceEngine engType;
    QInterfaceEngine subEngType;
    int devID;
    std::vector<int> devices;
    complex phaseFactor;
    bool useRDRAND;
    bool isSparse;
    bool useHostRam;
    bool isCacheEmpty;
    bitLenInt thresholdQubits;
    real1_f separabilityThreshold;
    std::vector<QMaskFusionShard> zxShards;

    QInterfacePtr MakeEngine(bitCapInt initState = 0);

    void FlushBuffers();
    void DumpBuffers()
    {
        isCacheEmpty = true;
        DumpBuffers(0, qubitCount);
    }

    void DumpBuffers(const bitLenInt start, const bitLenInt length)
    {
        bitLenInt maxLcv = start + length;
        for (bitLenInt i = start; i < maxLcv; i++) {
            DumpBuffer(i);
        }
    }

    void DumpBuffer(const bitLenInt target)
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

    bool FlushIfBuffered(const bitLenInt start, const bitLenInt length)
    {
        if (isCacheEmpty) {
            return true;
        }

        bitLenInt maxLcv = start + length;
        for (bitLenInt i = start; i < maxLcv; i++) {
            if (zxShards[i].isBuffered()) {
                FlushBuffers();
                return true;
            }
        }

        return false;
    }

    bool FlushIfPhaseBlocked() { return FlushIfPhaseBlocked((bitLenInt)0U, qubitCount); }

    bool FlushIfPhaseBlocked(const bitLenInt* controls, const bitLenInt controlLen)
    {
        if (isCacheEmpty) {
            return true;
        }

        bitLenInt control, i;
        bool isBlocked = false;
        for (i = 0U; i < controlLen; i++) {
            control = controls[i];
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

    bool FlushIfPhaseBlocked(const bitLenInt target)
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

    bool FlushIfPhaseBlocked(const bitLenInt start, const bitLenInt length)
    {
        if (isCacheEmpty) {
            return true;
        }

        bool isBlocked = false;
        bitLenInt maxLcv = start + length;
        for (bitLenInt i = start; i < maxLcv; i++) {
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
    QMaskFusion(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);
    QMaskFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QMaskFusion(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold, separation_thresh)
    {
    }
    QMaskFusion(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QMaskFusion(QINTERFACE_OPTIMAL_SCHROEDINGER, QINTERFACE_OPTIMAL_SINGLE_PAGE, qBitCount, initState, rgp,
              phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh,
              devList, qubitThreshold, separation_thresh)
    {
    }

    virtual real1_f ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
    {
        FlushIfPhaseBlocked(start, length);
        return engine->ProbReg(start, length, permutation);
    }

    using QInterface::Compose;
    virtual bitLenInt Compose(QMaskFusionPtr toCopy)
    {
        bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;
        zxShards.insert(zxShards.end(), toCopy->zxShards.begin(), toCopy->zxShards.end());
        SetQubitCount(nQubitCount);
        return engine->Compose(toCopy->engine);
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QMaskFusion>(toCopy)); }
    virtual bitLenInt Compose(QMaskFusionPtr toCopy, bitLenInt start)
    {
        bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;
        zxShards.insert(zxShards.begin() + start, toCopy->zxShards.begin(), toCopy->zxShards.end());
        SetQubitCount(nQubitCount);
        return engine->Compose(toCopy->engine, start);
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QMaskFusion>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QMaskFusion>(dest));
    }
    virtual bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QMaskFusion>(dest), error_tol);
    }
    virtual void Decompose(bitLenInt start, QMaskFusionPtr dest)
    {
        bitLenInt length = dest->GetQubitCount();
        bitLenInt nQubitCount = qubitCount - length;
        std::copy(zxShards.begin() + start, zxShards.begin() + start + length, dest->zxShards.begin());
        zxShards.erase(zxShards.begin() + start, zxShards.begin() + start + length);
        SetQubitCount(nQubitCount);
        return engine->Decompose(start, dest->engine);
    }
    virtual void Dispose(bitLenInt start, bitLenInt length)
    {
        bitLenInt nQubitCount = qubitCount - length;
        zxShards.erase(zxShards.begin() + start, zxShards.begin() + start + length);
        SetQubitCount(nQubitCount);
        return engine->Dispose(start, length);
    }
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        bitLenInt nQubitCount = qubitCount - length;
        zxShards.erase(zxShards.begin() + start, zxShards.begin() + start + length);
        SetQubitCount(nQubitCount);
        return engine->Dispose(start, length, disposedPerm);
    }
    virtual bool TryDecompose(bitLenInt start, QMaskFusionPtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
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

    virtual void SetQuantumState(const complex* inputState)
    {
        DumpBuffers();
        engine->SetQuantumState(inputState);
    }
    virtual void GetQuantumState(complex* outputState)
    {
        FlushBuffers();
        engine->GetQuantumState(outputState);
    }
    virtual void GetProbs(real1* outputProbs)
    {
        FlushIfPhaseBlocked();
        engine->GetProbs(outputProbs);
    }
    virtual complex GetAmplitude(bitCapInt perm)
    {
        FlushBuffers();
        return engine->GetAmplitude(perm);
    }
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        FlushBuffers();
        engine->SetAmplitude(perm, amp);
    }
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        DumpBuffers();
        engine->SetPermutation(perm, phaseFac);
    }

    using QInterface::X;
    virtual void X(bitLenInt target);
    using QInterface::Y;
    virtual void Y(bitLenInt target);
    using QInterface::Z;
    virtual void Z(bitLenInt target);

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt target);
    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target)
    {
        if (IS_SAME(topLeft, bottomRight)) {
            return;
        }

        if (IS_SAME(topLeft, -bottomRight)) {
            Z(target);
            return;
        }

        complex tl = topLeft;
        complex br = bottomRight;

        if (zxShards[target].isZ) {
            zxShards[target].isZ = false;
            br = -br;
        }

        if (zxShards[target].isX) {
            zxShards[target].isX = false;
            engine->ApplySingleInvert(tl, br, target);
        } else {
            engine->ApplySinglePhase(tl, br, target);
        }
    }
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target)
    {
        if (IS_SAME(topRight, bottomLeft)) {
            X(target);
            return;
        }

        if (IS_SAME(topRight, -bottomLeft)) {
            Y(target);
            return;
        }

        complex tr = topRight;
        complex bl = bottomLeft;

        if (zxShards[target].isZ) {
            zxShards[target].isZ = false;
            tr = -tr;
        }

        if (zxShards[target].isX) {
            zxShards[target].isX = false;
            engine->ApplySinglePhase(tr, bl, target);
        } else {
            engine->ApplySingleInvert(tr, bl, target);
        }
    }

    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
            ApplyControlledSinglePhase(controls, controlLen, target, mtrx[0], mtrx[3]);
            return;
        }

        if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
            ApplyControlledSingleInvert(controls, controlLen, target, mtrx[1], mtrx[2]);
            return;
        }

        FlushIfBuffered(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
    }
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
            ApplyAntiControlledSinglePhase(controls, controlLen, target, mtrx[0], mtrx[3]);
            return;
        }

        if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
            ApplyAntiControlledSingleInvert(controls, controlLen, target, mtrx[1], mtrx[2]);
            return;
        }

        FlushIfBuffered(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
    }
    virtual void ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight)
    {
        FlushIfPhaseBlocked(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->ApplyControlledSinglePhase(controls, controlLen, target, topLeft, bottomRight);
    }
    virtual void ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft)
    {
        FlushIfBuffered(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->ApplyControlledSingleInvert(controls, controlLen, target, topRight, bottomLeft);
    }
    virtual void ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight)
    {
        FlushIfPhaseBlocked(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->ApplyAntiControlledSinglePhase(controls, controlLen, target, topLeft, bottomRight);
    }
    virtual void ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft)
    {
        FlushIfBuffered(target) || FlushIfPhaseBlocked(controls, controlLen);
        engine->ApplyAntiControlledSingleInvert(controls, controlLen, target, topRight, bottomLeft);
    }

    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask)
    {
        FlushIfBuffered(qubitIndex) || FlushIfPhaseBlocked(controls, controlLen);
        engine->UniformlyControlledSingleBit(
            controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
    }

    virtual void UniformParityRZ(const bitCapInt& mask, const real1_f& angle) { engine->UniformParityRZ(mask, angle); }
    virtual void CUniformParityRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle)
    {
        FlushBuffers();
        engine->CUniformParityRZ(controls, controlLen, mask, angle);
    }

    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->CSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->AntiCSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->CSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->CISqrtSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        FlushIfBuffered(qubit1) || FlushIfBuffered(qubit2) || FlushIfPhaseBlocked(controls, controlLen);
        engine->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2);
    }

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        FlushIfPhaseBlocked(qubit);
        DumpBuffer(qubit);
        return engine->ForceM(qubit, result, doForce, doApply);
    }

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        FlushIfBuffered(start, length);
        engine->INC(toAdd, start, length);
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfPhaseBlocked(controls, controlLen);
        engine->CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(carryIndex);
        engine->INCC(toAdd, start, length, carryIndex);
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(overflowIndex);
        engine->INCS(toAdd, start, length, overflowIndex);
    }
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(overflowIndex) || FlushIfBuffered(carryIndex);
        engine->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(carryIndex);
        engine->INCSC(toAdd, start, length, carryIndex);
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(carryIndex);
        engine->DECC(toSub, start, length, carryIndex);
    }
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(overflowIndex) || FlushIfBuffered(carryIndex);
        engine->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(carryIndex);
        engine->DECSC(toSub, start, length, carryIndex);
    }
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        FlushIfBuffered(start, length);
        engine->INCBCD(toAdd, start, length);
    }
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(carryIndex);
        engine->INCBCDC(toAdd, start, length, carryIndex);
    }
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(carryIndex);
        engine->DECBCDC(toSub, start, length, carryIndex);
    }
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryStart, length);
        engine->MUL(toMul, inOutStart, carryStart, length);
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryStart, length);
        engine->DIV(toDiv, inOutStart, carryStart, length);
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length);
        engine->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length);
        engine->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length);
        engine->POWModNOut(base, modN, inStart, outStart, length);
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inOutStart, length) || FlushIfBuffered(carryStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        FlushIfBuffered(inStart, length) || FlushIfBuffered(outStart, length) ||
            FlushIfPhaseBlocked(controls, controlLen);
        engine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
    }

    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        FlushIfBuffered(start, length) || FlushIfBuffered(flagIndex);
        engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        FlushIfBuffered(start, length);
        engine->PhaseFlipIfLess(greaterPerm, start, length);
    }

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true)
    {
        FlushIfBuffered(indexStart, indexLength) || FlushIfBuffered(valueStart, valueLength);
        return engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        FlushIfBuffered(indexStart, indexLength) || FlushIfBuffered(valueStart, valueLength) ||
            FlushIfBuffered(carryIndex);
        return engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        FlushIfBuffered(indexStart, indexLength) || FlushIfBuffered(valueStart, valueLength) ||
            FlushIfBuffered(carryIndex);
        return engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values)
    {
        FlushIfBuffered(start, length);
        engine->Hash(start, length, values);
    }

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        std::swap(zxShards[qubitIndex1], zxShards[qubitIndex2]);
        engine->Swap(qubitIndex1, qubitIndex2);
    }
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->ISwap(qubitIndex1, qubitIndex2);
    }
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
    }
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        FlushIfBuffered(qubitIndex1) || FlushIfBuffered(qubitIndex2);
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    virtual real1_f Prob(bitLenInt qubitIndex)
    {
        if (zxShards[qubitIndex].isX) {
            return clampProb(ONE_R1 - engine->Prob(qubitIndex));
        }

        return engine->Prob(qubitIndex);
    }
    virtual real1_f ProbAll(bitCapInt fullRegister)
    {
        FlushIfPhaseBlocked();
        return engine->ProbAll(fullRegister);
    }
    virtual real1_f ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        FlushBuffers();
        return engine->ProbMask(mask, permutation);
    }
    virtual real1_f ProbParity(const bitCapInt& mask)
    {
        FlushBuffers();
        return engine->ProbParity(mask);
    }
    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true)
    {
        FlushBuffers();
        return engine->ForceMParity(mask, result, doForce);
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QMaskFusion>(toCompare));
    }
    virtual real1_f SumSqrDiff(QMaskFusionPtr toCompare)
    {
        FlushBuffers();
        toCompare->FlushBuffers();
        return engine->SumSqrDiff(toCompare->engine);
    }

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) { engine->UpdateRunningNorm(norm_thresh); }
    virtual void NormalizeState(real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        engine->NormalizeState(nrm, norm_thresh);
    }

    virtual real1_f ExpectationBitsAll(const bitLenInt* bits, const bitLenInt& length, const bitCapInt& offset = 0)
    {
        FlushIfPhaseBlocked(bits, length);
        return engine->ExpectationBitsAll(bits, length, offset);
    }

    virtual bool TrySeparate(bitLenInt qubit) { return engine->TrySeparate(qubit); }
    virtual bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2) { return engine->TrySeparate(qubit1, qubit2); }
    virtual bool TrySeparate(bitLenInt* qubits, bitLenInt length, real1_f error_tol)
    {
        return engine->TrySeparate(qubits, length, error_tol);
    }

    virtual void Finish() { engine->Finish(); }

    virtual bool isFinished() { return engine->isFinished(); }

    virtual QInterfacePtr Clone();

    virtual void SetDevice(const int& dID, const bool& forceReInit = false)
    {
        devID = dID;
        engine->SetDevice(dID, forceReInit);
    }

    virtual int GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize() { return engine->GetMaxSize(); };
};
} // namespace Qrack
