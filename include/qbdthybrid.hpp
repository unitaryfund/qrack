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

#include "qinterface.hpp"

namespace Qrack {

class QBdtHybrid;
typedef std::shared_ptr<QBdtHybrid> QBdtHybridPtr;

/**
 * A "Qrack::QBdtHybrid" internally switched between Qrack::QBdt and Qrack::QHybrid to maximize
 * entanglement-dependent performance.
 */
#if ENABLE_ALU
class QBdtHybrid : public QAlu, public QParity, public QInterface {
#else
class QBdtHybrid : public QParity, public QInterface {
#endif
protected:
    bool useRDRAND;
    bool isSparse;
    bool useHostRam;
    bitLenInt thresholdQubits;
    real1_f separabilityThreshold;
    int64_t devID;
    QBdtPtr qbdt;
    QEnginePtr engine;
    complex phaseFactor;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engines;

    /**
     * Switches between QBdt and QEngine modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically after every gate, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    void SwitchMode(bool useBdt)
    {
        if (!engine != useBdt) {
            QInterfacePtr nEngine = MakeSimulator(useBdt);
            std::unique_ptr<complex> sv(new complex[(size_t)maxQPower]);
            qbdt->GetQuantumState(sv.get());
            nEngine->SetQuantumState(sv.get());
            if (useBdt) {
                qbdt = std::dynamic_pointer_cast<QBdt>(nEngine);
                engine = NULL;
            } else {
                qbdt = NULL;
                engine = std::dynamic_pointer_cast<QEngine>(nEngine);
            }
        }
    }

public:
    QBdtHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0U,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QBdtHybrid(bitLenInt qBitCount, bitCapInt initState = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QBdtHybrid({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    QInterfacePtr MakeSimulator(bool isBdt);

    bool isBinaryDecisionTree() { return !engine; }

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        if (qbdt) {
            qbdt->SetConcurrency(GetConcurrencyLevel());
        } else {
            engine->SetConcurrency(GetConcurrencyLevel());
        }
    }

    real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
    {
        if (qbdt) {
            return qbdt->ProbReg(start, length, permutation);
        } else {
            return engine->ProbReg(start, length, permutation);
        }
    }

    using QInterface::Compose;
    bitLenInt Compose(QBdtHybridPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchMode(!engine);
        if (qbdt) {
            return qbdt->Compose(toCopy->qbdt);
        } else {
            return engine->Compose(toCopy->engine);
        }
        // TODO: After every relevant if/else, QBdt::CountBranches();
    }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QBdtHybrid>(toCopy)); }
    bitLenInt Compose(QBdtHybridPtr toCopy, bitLenInt start)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchMode(!engine);
        if (qbdt) {
            return qbdt->Compose(toCopy->qbdt, start);
        } else {
            return engine->Compose(toCopy->engine, start);
        }
    }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QBdtHybrid>(toCopy), start);
    }
    bitLenInt ComposeNoClone(QBdtHybridPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        toCopy->SwitchMode(!engine);
        if (qbdt) {
            return qbdt->ComposeNoClone(toCopy->qbdt);
        } else {
            return engine->ComposeNoClone(toCopy->engine);
        }
    }
    bitLenInt ComposeNoClone(QInterfacePtr toCopy)
    {
        return ComposeNoClone(std::dynamic_pointer_cast<QBdtHybrid>(toCopy));
    }
    using QInterface::Decompose;
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QBdtHybrid>(dest));
    }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length)
    {
        if (qbdt) {
            return qbdt->Decompose(start, length);
        } else {
            return engine->Decompose(start, length);
        }
    }
    bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QBdtHybrid>(dest), error_tol);
    }
    void Decompose(bitLenInt start, QBdtHybridPtr dest)
    {
        dest->SwitchMode(!engine);
        if (qbdt) {
            qbdt->Decompose(start, dest->qbdt);
        } else {
            engine->Decompose(start, dest->qbdt);
        }
        SetQubitCount(qubitCount - dest->GetQubitCount());
    }
    void Dispose(bitLenInt start, bitLenInt length)
    {
        if (qbdt) {
            qbdt->Dispose(start, length);
        } else {
            engine->Dispose(start, length);
        }
        SetQubitCount(qubitCount - length);
    }
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        if (qbdt) {
            qbdt->Dispose(start, length, disposedPerm);
        } else {
            engine->Dispose(start, length, disposedPerm);
        }
        SetQubitCount(qubitCount - length);
    }

    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length)
    {
        if (!length) {
            return start;
        }

        QBdtHybridPtr nQubits = std::make_shared<QBdtHybrid>(length, 0U, rand_generator, phaseFactor, doNormalize,
            randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
            thresholdQubits, separabilityThreshold);
        nQubits->SetConcurrency(GetConcurrencyLevel());

        return Compose(nQubits, start);
    }

    void SetQuantumState(const complex* inputState)
    {
        if (qbdt) {
            qbdt->SetQuantumState(inputState);
        } else {
            engine->SetQuantumState(inputState);
        }
    }
    void GetQuantumState(complex* outputState)
    {
        if (qbdt) {
            qbdt->GetQuantumState(outputState);
        } else {
            engine->GetQuantumState(outputState);
        }
    }
    void GetProbs(real1* outputProbs)
    {
        if (qbdt) {
            qbdt->GetProbs(outputProbs);
        } else {
            engine->GetProbs(outputProbs);
        }
    }
    complex GetAmplitude(bitCapInt perm)
    {
        if (qbdt) {
            return qbdt->GetAmplitude(perm);
        } else {
            return engine->GetAmplitude(perm);
        }
    }
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        if (qbdt) {
            qbdt->SetAmplitude(perm, amp);
        } else {
            engine->SetAmplitude(perm, amp);
        }
    }
    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        if (qbdt) {
            qbdt->SetPermutation(perm, phaseFac);
        } else {
            engine->SetPermutation(perm, phaseFac);
        }
    }

    void Mtrx(const complex* mtrx, bitLenInt qubitIndex)
    {
        if (qbdt) {
            qbdt->Mtrx(mtrx, qubitIndex);
        } else {
            engine->Mtrx(mtrx, qubitIndex);
        }
    }
    void Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex)
    {
        if (qbdt) {
            qbdt->Phase(topLeft, bottomRight, qubitIndex);
        } else {
            engine->Phase(topLeft, bottomRight, qubitIndex);
        }
    }
    void Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex)
    {
        if (qbdt) {
            qbdt->Invert(topRight, bottomLeft, qubitIndex);
        } else {
            engine->Invert(topRight, bottomLeft, qubitIndex);
        }
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        if (qbdt) {
            qbdt->MCMtrx(controls, mtrx, target);
        } else {
            engine->MCMtrx(controls, mtrx, target);
        }
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        if (qbdt) {
            qbdt->MACMtrx(controls, mtrx, target);
        } else {
            engine->MACMtrx(controls, mtrx, target);
        }
    }

    using QInterface::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
        const complex* mtrxs, const std::vector<bitCapInt> mtrxSkipPowers, bitCapInt mtrxSkipValueMask)
    {
        if (qbdt) {
            qbdt->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipValueMask);
        } else {
            engine->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipValueMask);
        }
    }

    void XMask(bitCapInt mask)
    {
        if (qbdt) {
            qbdt->XMask(mask);
        } else {
            engine->XMask(mask);
        }
    }
    void PhaseParity(real1_f radians, bitCapInt mask)
    {
        if (qbdt) {
            qbdt->PhaseParity(radians, mask);
        } else {
            engine->PhaseParity(radians, mask);
        }
    }

    real1_f CProb(bitLenInt control, bitLenInt target) { return qbdt->CProb(control, target); }
    real1_f ACProb(bitLenInt control, bitLenInt target) { return qbdt->ACProb(control, target); }

    void UniformParityRZ(bitCapInt mask, real1_f angle) { qbdt->UniformParityRZ(mask, angle); }
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, bitCapInt mask, real1_f angle)
    {
        qbdt->CUniformParityRZ(controls, mask, angle);
    }

    void CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        qbdt->CSwap(controls, qubit1, qubit2);
    }
    void AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        qbdt->AntiCSwap(controls, qubit1, qubit2);
    }
    void CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        qbdt->CSqrtSwap(controls, qubit1, qubit2);
    }
    void AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        qbdt->AntiCSqrtSwap(controls, qubit1, qubit2);
    }
    void CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        qbdt->CISqrtSwap(controls, qubit1, qubit2);
    }
    void AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        qbdt->AntiCISqrtSwap(controls, qubit1, qubit2);
    }

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        return qbdt->ForceM(qubit, result, doForce, doApply);
    }

#if ENABLE_ALU
    using QInterface::M;
    bool M(bitLenInt q) { return QInterface::M(q); }
    using QInterface::X;
    void X(bitLenInt q) { QInterface::X(q); }
    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) { QInterface::INC(toAdd, start, length); }
    void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) { QInterface::DEC(toSub, start, length); }
    void CDEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        QInterface::CDEC(toSub, inOutStart, length, controls);
    }
    void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCDECC(toAdd, start, length, carryIndex);
    }
    void CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        qbdt->CINC(toAdd, inOutStart, length, controls);
    }
    void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qbdt->INCC(toAdd, start, length, carryIndex);
    }
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        qbdt->INCS(toAdd, start, length, overflowIndex);
    }
    void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        qbdt->DECS(toAdd, start, length, overflowIndex);
    }
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        qbdt->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qbdt->INCSC(toAdd, start, length, carryIndex);
    }
    void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qbdt->DECC(toSub, start, length, carryIndex);
    }
    void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        qbdt->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }
    void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qbdt->DECSC(toSub, start, length, carryIndex);
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        qbdt->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qbdt->INCDECSC(toAdd, start, length, carryIndex);
    }
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) { qbdt->INCBCD(toAdd, start, length); }
    void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qbdt->INCBCDC(toAdd, start, length, carryIndex);
    }
    void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        qbdt->DECBCDC(toSub, start, length, carryIndex);
    }
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        qbdt->MUL(toMul, inOutStart, carryStart, length);
    }
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        qbdt->DIV(toDiv, inOutStart, carryStart, length);
    }
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qbdt->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qbdt->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qbdt->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        qbdt->CMUL(toMul, inOutStart, carryStart, length, controls);
    }
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        qbdt->CDIV(toDiv, inOutStart, carryStart, length, controls);
    }
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        qbdt->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        qbdt->CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        qbdt->CPOWModNOut(base, modN, inStart, outStart, length, controls);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        return qbdt->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        return qbdt->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        return qbdt->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values) { qbdt->Hash(start, length, values); }

    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        qbdt->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        qbdt->PhaseFlipIfLess(greaterPerm, start, length);
    }
#endif

    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { qbdt->Swap(qubitIndex1, qubitIndex2); }
    void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { qbdt->ISwap(qubitIndex1, qubitIndex2); }
    void IISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { qbdt->IISwap(qubitIndex1, qubitIndex2); }
    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { qbdt->SqrtSwap(qubitIndex1, qubitIndex2); }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { qbdt->ISqrtSwap(qubitIndex1, qubitIndex2); }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        qbdt->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    real1_f Prob(bitLenInt qubitIndex) { return qbdt->Prob(qubitIndex); }
    real1_f ProbAll(bitCapInt fullRegister) { return qbdt->ProbAll(fullRegister); }
    real1_f ProbMask(bitCapInt mask, bitCapInt permutation) { return qbdt->ProbMask(mask, permutation); }
    real1_f ProbParity(bitCapInt mask) { return qbdt->ProbParity(mask); }
    bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
    {
        return qbdt->ForceMParity(mask, result, doForce);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare) { return SumSqrDiff(std::dynamic_pointer_cast<QBdtHybrid>(toCompare)); }
    real1_f SumSqrDiff(QBdtHybridPtr toCompare)
    {
        toCompare->SwitchMode(!engine);
        return qbdt->SumSqrDiff(toCompare->qbdt);
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) { qbdt->UpdateRunningNorm(norm_thresh); }
    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        qbdt->NormalizeState(nrm, norm_thresh, phaseArg);
    }

    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, bitCapInt offset = 0)
    {
        return qbdt->ExpectationBitsAll(bits, offset);
    }

    void Finish() { qbdt->Finish(); }

    bool isFinished() { return qbdt->isFinished(); }

    void Dump() { qbdt->Dump(); }

    QInterfacePtr Clone()
    {
        QBdtHybridPtr c = std::make_shared<QBdtHybrid>(qubitCount, 0U, rand_generator, phaseFactor, doNormalize,
            randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, deviceIDs,
            thresholdQubits, separabilityThreshold);
        c->SetConcurrency(GetConcurrencyLevel());
        if (!engine) {
            c->qbdt->SetRoot(qbdt->GetRoot()->ShallowClone());
        } else {
            c->engine->CopyStateVec(engine);
        }

        return c;
    }

    void SetDevice(int64_t dID)
    {
        devID = dID;
        qbdt->SetDevice(dID);
    }

    int64_t GetDevice() { return devID; }

    bitCapIntOcl GetMaxSize() { return qbdt->GetMaxSize(); };
};
} // namespace Qrack
