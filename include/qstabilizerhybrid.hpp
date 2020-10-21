//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
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

class QStabilizerHybrid;
typedef std::shared_ptr<QStabilizerHybrid> QStabilizerHybridPtr;

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
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1 norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0);

    QStabilizerHybrid(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1 norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0)
        : QStabilizerHybrid(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem,
              deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold)
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

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    virtual void SwitchToEngine();

    virtual bool isClifford() { return !engine; }

    /// Apply a CNOT gate with control and target
    virtual void CNOT(bitLenInt control, bitLenInt target)
    {
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
        if (stabilizer) {
            stabilizer->Z(target);
        } else {
            engine->Z(target);
        }
    }

    virtual void IS(bitLenInt target)
    {
        if (stabilizer) {
            stabilizer->IS(target);
        } else {
            engine->IS(target);
        }
    }

    virtual void CIS(bitLenInt control, bitLenInt target);

    virtual void X(bitLenInt target)
    {
        if (stabilizer) {
            stabilizer->X(target);
        } else {
            engine->X(target);
        }
    }

    virtual void Y(bitLenInt target)
    {
        if (stabilizer) {
            stabilizer->Y(target);
        } else {
            engine->Y(target);
        }
    }

    virtual void CZ(bitLenInt control, bitLenInt target)
    {
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
        if (controlLen == 0) {
            ApplySingleBit(mtrxs, qubitIndex);
            return;
        }

        SwitchToEngine();
        engine->UniformlyControlledSingleBit(
            controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
    }

    virtual void UniformParityRZ(const bitCapInt& mask, const real1& angle)
    {
        SwitchToEngine();
        engine->UniformParityRZ(mask, angle);
    }

    virtual void CUniformParityRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1& angle)
    {
        SwitchToEngine();
        engine->CUniformParityRZ(controls, controlLen, mask, angle);
    }

    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        if (controlLen == 0) {
            Swap(qubit1, qubit2);
            return;
        }

        SwitchToEngine();
        engine->CSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        if (controlLen == 0) {
            Swap(qubit1, qubit2);
            return;
        }

        SwitchToEngine();
        engine->AntiCSwap(controls, controlLen, qubit1, qubit2);
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
        // TODO: QStabilizer appears not to be decomposable after measurement, and in many cases where a bit is in an
        // eigenstate.
        if (stabilizer && ((engineType == QINTERFACE_QUNIT) || (engineType == QINTERFACE_QUNIT_MULTI))) {
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
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        engine->DECBCDC(toSub, start, length, carryIndex);
    }
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
    virtual void FSim(real1 theta, real1 phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        SwitchToEngine();
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    virtual real1 Prob(bitLenInt qubitIndex)
    {
        if (engine) {
            return engine->Prob(qubitIndex);
        }

        if (stabilizer->IsSeparableZ(qubitIndex)) {
            return stabilizer->M(qubitIndex) ? ONE_R1 : ZERO_R1;
        } else {
            return ONE_R1 / 2;
        }
    }
    virtual real1 ProbAll(bitCapInt fullRegister)
    {
        SwitchToEngine();
        return engine->ProbAll(fullRegister);
    }
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        SwitchToEngine();
        return engine->ProbMask(mask, permutation);
    }
    // TODO: Good opportunity to optimize
    virtual real1 ProbParity(const bitCapInt& mask)
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

    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare));
    }

    virtual bool ApproxCompare(QStabilizerHybridPtr toCompare)
    {
        if (!stabilizer == !(toCompare->engine)) {
            return false;
        }

        if (stabilizer) {
            return stabilizer->ApproxCompare(toCompare->stabilizer);
        }

        return engine->ApproxCompare(toCompare->engine);
    }
    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->UpdateRunningNorm(norm_thresh);
        }
    }
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->NormalizeState(nrm, norm_thresh);
        }
    }

    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1)
    {
        // if (stabilizer) {
        //     if (length == 1) {
        //         return stabilizer->IsSeparable(start);
        //     }
        // }

        if (stabilizer) {
            return false;
        }

        return engine->TrySeparate(start, length);
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
