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

public:
    QStabilizerHybrid(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0);

    QStabilizerPtr MakeStabilizer(const bitCapInt& perm);
    QInterfacePtr MakeEngine();

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
    virtual void SwitchToEngine()
    {
        if (engine) {
            return;
        }

        complex* stateVec = new complex[maxQPower];
        stabilizer->GetQuantumState(stateVec);
        stabilizer = NULL;

        engine = MakeEngine();
        engine->SetQuantumState(stateVec);
        delete[] stateVec;
    }

    /// Apply a CNOT gate with control and target
    virtual void CNOT(const bitLenInt& control, const bitLenInt& target)
    {
        if (stabilizer) {
            stabilizer->CNOT(control, target);
        } else {
            engine->CNOT(control, target);
        }
    }

    /// Apply a Hadamard gate to target
    virtual void H(const bitLenInt& target)
    {
        if (stabilizer) {
            stabilizer->H(target);
        } else {
            engine->H(target);
        }
    }

    /// Apply a phase gate (|0>->|0>, |1>->i|1>, or "S") to qubit b
    virtual void S(const bitLenInt& target)
    {
        if (stabilizer) {
            stabilizer->S(target);
        } else {
            engine->S(target);
        }
    }

    // TODO: Custom implementations for decompositions:
    virtual void Z(const bitLenInt& target)
    {
        if (stabilizer) {
            stabilizer->Z(target);
        } else {
            engine->Z(target);
        }
    }

    virtual void IS(const bitLenInt& target)
    {
        if (stabilizer) {
            stabilizer->IS(target);
        } else {
            engine->IS(target);
        }
    }

    virtual void X(const bitLenInt& target)
    {
        if (stabilizer) {
            stabilizer->X(target);
        } else {
            engine->X(target);
        }
    }

    virtual void Y(const bitLenInt& target)
    {
        if (stabilizer) {
            stabilizer->Y(target);
        } else {
            engine->Y(target);
        }
    }

    virtual void CZ(const bitLenInt& control, const bitLenInt& target)
    {
        if (stabilizer) {
            stabilizer->CZ(control, target);
        } else {
            engine->CZ(control, target);
        }
    }

    virtual void Swap(const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        if (stabilizer) {
            stabilizer->Swap(qubit1, qubit2);
        } else {
            engine->Swap(qubit1, qubit2);
        }
    }

    virtual void ISwap(const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        if (stabilizer) {
            stabilizer->ISwap(qubit1, qubit2);
        } else {
            engine->ISwap(qubit1, qubit2);
        }
    }

    using QInterface::Compose;
    virtual bitLenInt Compose(QStabilizerHybridPtr toCopy)
    {
        if (engine) {
            if (toCopy->stabilizer) {
                toCopy->SwitchToEngine();
            }
            return engine->Compose(toCopy->engine);
        }

        if (toCopy->engine) {
            SwitchToEngine();
            return engine->Compose(toCopy->engine);
        }

        return stabilizer->Compose(toCopy->stabilizer);
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy));
    }
    virtual bitLenInt Compose(QStabilizerHybridPtr toCopy, bitLenInt start)
    {
        if (engine) {
            if (toCopy->stabilizer) {
                toCopy->SwitchToEngine();
            }
            return engine->Compose(toCopy->engine, start);
        }

        if (toCopy->engine) {
            SwitchToEngine();
            return engine->Compose(toCopy->engine, start);
        }

        return stabilizer->Compose(toCopy->stabilizer, start);
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        Decompose(start, length, std::dynamic_pointer_cast<QStabilizerHybrid>(dest));
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QStabilizerHybridPtr dest)
    {
        if (engine) {
            if (dest->stabilizer) {
                dest->SwitchToEngine();
            }
            return engine->Decompose(start, length, dest->engine);
        }

        if (dest->engine) {
            SwitchToEngine();
            return engine->Decompose(start, length, dest->engine);
        }

        return stabilizer->Decompose(start, length, dest->stabilizer);
    }
    virtual void Dispose(bitLenInt start, bitLenInt length)
    {
        if (engine) {
            engine->Dispose(start, length);
        } else if (stabilizer) {
            stabilizer->Dispose(start, length);
        }
    }
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        if (engine) {
            engine->Dispose(start, length, disposedPerm);
        } else if (stabilizer) {
            stabilizer->Dispose(start, length);
        }
    }

    virtual void SetQuantumState(const complex* inputState)
    {
        SwitchToEngine();
        engine->SetQuantumState(inputState);
    }
    virtual void GetQuantumState(complex* outputState)
    {
        if (stabilizer) {
            stabilizer->GetQuantumState(outputState);
        } else {
            engine->GetQuantumState(outputState);
        }
    }
    virtual void GetProbs(real1* outputProbs)
    {
        if (stabilizer) {
            complex* stateVec = new complex[maxQPower];
            stabilizer->GetQuantumState(stateVec);
            for (bitCapInt i = 0; i < maxQPower; i++) {
                outputProbs[i] = norm(stateVec[i]);
            }
            delete[] stateVec;
        } else {
            engine->GetProbs(outputProbs);
        }
    }
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

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt target)
    {
        if (IsIdentity(mtrx, true)) {
            return;
        }

        if ((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) {
            ApplySinglePhase(mtrx[0], mtrx[3], target);
            return;
        }
        if ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)) {
            ApplySingleInvert(mtrx[1], mtrx[2], target);
            return;
        }
        if ((mtrx[0] == complex(M_SQRT1_2, ZERO_R1)) && (mtrx[0] == mtrx[1]) && (mtrx[0] == mtrx[2]) &&
            (mtrx[2] == -mtrx[3])) {
            H(target);
            return;
        }

        SwitchToEngine();
        engine->ApplySingleBit(mtrx, target);
    }

    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target)
    {
        if (engine) {
            engine->ApplySinglePhase(topLeft, bottomRight, target);
            return;
        }

        if (topLeft == bottomRight) {
            return;
        }

        if (topLeft == -bottomRight) {
            stabilizer->Z(target);
            return;
        }

        complex sTest = bottomRight / topLeft;

        if (sTest == I_CMPLX) {
            stabilizer->S(target);
            return;
        }

        if (sTest == -I_CMPLX) {
            stabilizer->IS(target);
            return;
        }

        SwitchToEngine();
        engine->ApplySinglePhase(topLeft, bottomRight, target);
    }

    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target)
    {
        if (engine) {
            engine->ApplySingleInvert(topRight, bottomLeft, target);
            return;
        }

        if (topRight == bottomLeft) {
            stabilizer->X(target);
            return;
        }

        if (topRight == -bottomLeft) {
            stabilizer->Z(target);
            stabilizer->X(target);
            return;
        }

        complex sTest = bottomLeft / topRight;

        if (sTest == I_CMPLX) {
            stabilizer->S(target);
            stabilizer->X(target);
            return;
        }

        if (sTest == -I_CMPLX) {
            stabilizer->IS(target);
            stabilizer->X(target);
            return;
        }

        SwitchToEngine();
        engine->ApplySingleInvert(topRight, bottomLeft, target);
    }

    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        if (controlLen > 1U) {
            SwitchToEngine();
        }

        if (engine) {
            engine->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
            return;
        }

        if ((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) {
            ApplyControlledSinglePhase(controls, controlLen, target, mtrx[0], mtrx[3]);
            return;
        }
        if ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)) {
            ApplyControlledSingleInvert(controls, controlLen, target, mtrx[1], mtrx[2]);
            return;
        }

        SwitchToEngine();
        engine->ApplyControlledSingleBit(controls, controlLen, target, mtrx);
    }

    virtual void ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight)
    {
        if (!controlLen) {
            ApplySinglePhase(topLeft, bottomRight, target);
            return;
        }

        if ((topLeft != ONE_CMPLX) || (controlLen > 1U)) {
            SwitchToEngine();
        }

        if (engine) {
            engine->ApplyControlledSinglePhase(controls, controlLen, target, topLeft, bottomRight);
            return;
        }

        if (bottomRight == ONE_CMPLX) {
            return;
        }

        if (bottomRight == -ONE_CMPLX) {
            stabilizer->CZ(controls[0], target);
            return;
        }

        SwitchToEngine();
        engine->ApplyControlledSinglePhase(controls, controlLen, target, topLeft, bottomRight);
    }

    virtual void ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft)
    {
        if (!controlLen) {
            ApplySingleInvert(topRight, bottomLeft, target);
            return;
        }

        if ((topRight != ONE_CMPLX) || (controlLen > 1U)) {
            SwitchToEngine();
        }

        if (engine) {
            engine->ApplyControlledSingleInvert(controls, controlLen, target, topRight, bottomLeft);
            return;
        }

        if (bottomLeft == ONE_CMPLX) {
            stabilizer->CNOT(controls[0], target);
            return;
        }

        if (bottomLeft == -ONE_CMPLX) {
            stabilizer->CNOT(controls[0], target);
            stabilizer->CZ(controls[0], target);
            return;
        }

        SwitchToEngine();
        engine->ApplyControlledSingleInvert(controls, controlLen, target, topRight, bottomLeft);
    }

    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        if (controlLen > 1U) {
            SwitchToEngine();
        }

        if (engine) {
            engine->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
            return;
        }

        if ((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) {
            ApplyAntiControlledSinglePhase(controls, controlLen, target, mtrx[0], mtrx[3]);
            return;
        }
        if ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)) {
            ApplyAntiControlledSingleInvert(controls, controlLen, target, mtrx[1], mtrx[2]);
            return;
        }

        SwitchToEngine();
        engine->ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
    }

    virtual void ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight)
    {
        if (!controlLen) {
            ApplySinglePhase(topLeft, bottomRight, target);
            return;
        }

        if ((topLeft != ONE_CMPLX) || (controlLen > 1U)) {
            SwitchToEngine();
        }

        if (engine) {
            engine->ApplyAntiControlledSinglePhase(controls, controlLen, target, topLeft, bottomRight);
            return;
        }

        if (bottomRight == ONE_CMPLX) {
            return;
        }

        if (bottomRight == -ONE_CMPLX) {
            stabilizer->X(controls[0]);
            stabilizer->CZ(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }

        SwitchToEngine();
        engine->ApplyControlledSinglePhase(controls, controlLen, target, topLeft, bottomRight);
    }

    virtual void ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft)
    {
        if (!controlLen) {
            ApplySingleInvert(topRight, bottomLeft, target);
            return;
        }

        if ((topRight != ONE_CMPLX) || (controlLen > 1U)) {
            SwitchToEngine();
        }

        if (engine) {
            engine->ApplyAntiControlledSingleInvert(controls, controlLen, target, topRight, bottomLeft);
            return;
        }

        if (bottomLeft == ONE_CMPLX) {
            stabilizer->X(controls[0]);
            stabilizer->CNOT(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }

        if (bottomLeft == -ONE_CMPLX) {
            stabilizer->X(controls[0]);
            stabilizer->CZ(controls[0], target);
            stabilizer->X(controls[0]);
            return;
        }

        SwitchToEngine();
        engine->ApplyAntiControlledSingleInvert(controls, controlLen, target, topRight, bottomLeft);
    }

    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask)
    {
        SwitchToEngine();
        engine->UniformlyControlledSingleBit(
            controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
    }

    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        SwitchToEngine();
        engine->CSwap(controls, controlLen, qubit1, qubit2);
    }
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
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
        if (doApply == false) {
            SwitchToEngine();
        }

        if (stabilizer) {
            return stabilizer->M(qubit, doForce, result);
        }

        return engine->ForceM(qubit, result, doForce, doApply);
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
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) { engine->INCBCD(toAdd, start, length); }
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

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            stabilizer->Swap(qubitIndex1, qubitIndex2);
        } else {
            engine->Swap(qubitIndex1, qubitIndex2);
        }
    }
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            stabilizer->ISwap(qubitIndex1, qubitIndex2);
        } else {
            engine->ISwap(qubitIndex1, qubitIndex2);
        }
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
            return stabilizer->M(qubitIndex);
        } else {
            return ONE_R1 / 2;
        }
    }
    virtual real1 ProbAll(bitCapInt fullRegister)
    {
        if (engine) {
            return engine->ProbAll(fullRegister);
        }

        real1 prob = ONE_R1;
        for (bitLenInt i = 0; i < qubitCount; i++) {
            if (stabilizer->IsSeparableZ(i)) {
                if ((fullRegister & pow2(i)) != stabilizer->M(i)) {
                    return ZERO_R1;
                }
                // else, preserves full remaining probability
            } else {
                prob /= 2;
            }
        }

        return prob;
    }
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        if (engine) {
            return engine->ProbMask(mask, permutation);
        }

        bitCapInt pw;
        real1 prob = ONE_R1;
        for (bitLenInt i = 0; i < qubitCount; i++) {
            pw = pow2(i);
            if (!(mask && pw)) {
                continue;
            }

            if (stabilizer->IsSeparableZ(i)) {
                if ((permutation & pow2(i)) != stabilizer->M(i)) {
                    return ZERO_R1;
                }
                // else, preserves full remaining probability
            } else {
                prob /= 2;
            }
        }

        return prob;
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

    virtual void Finish()
    {
        if (engine) {
            engine->Finish();
        }
    }

    virtual bool isFinished()
    {
        if (stabilizer) {
            return true;
        }

        return engine->isFinished();
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
    };
};
} // namespace Qrack
