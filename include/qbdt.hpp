//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "mpsshard.hpp"
#include "qbdt_qinterface_node.hpp"
#include "qinterface.hpp"

namespace Qrack {

class QBdt;
typedef std::shared_ptr<QBdt> QBdtPtr;

class QBdt : virtual public QInterface {
protected:
    std::vector<QInterfaceEngine> engines;
    int devID;
    QBdtNodeInterfacePtr root;
    QInterfacePtr stateVecUnit;
    bitLenInt attachedQubitCount;
    bitLenInt bdtQubitCount;
    bitCapInt bdtMaxQPower;

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        bdtQubitCount = qubitCount - attachedQubitCount;
        bdtMaxQPower = pow2(bdtQubitCount);
    }

    QInterfacePtr MakeStateVector(bitLenInt qbCount, bitCapInt perm = 0U);
    QBdtQEngineNodePtr MakeQEngineNode(complex scale, bitLenInt qbCount, bitCapInt perm = 0U);

    QInterfacePtr MakeTempStateVector()
    {
        QInterfacePtr copyPtr = MakeStateVector(qubitCount);
        Finish();
        GetQuantumState(copyPtr);

        // If the calling function fully deferences our return, it's automatically freed.
        return copyPtr;
    }
    void SetStateVector()
    {
        if (stateVecUnit) {
            return;
        }

        Finish();
        stateVecUnit = MakeStateVector(qubitCount);
        GetQuantumState(stateVecUnit);
        root = NULL;
    }
    void ResetStateVector()
    {
        if (!stateVecUnit) {
            return;
        }

        SetQuantumState(stateVecUnit);
        stateVecUnit = NULL;
    }

    template <typename Fn> void GetTraversal(Fn getLambda);
    template <typename Fn> void SetTraversal(Fn setLambda);
    template <typename Fn> void ExecuteAsStateVector(Fn operation)
    {
        SetStateVector();
        operation(stateVecUnit);
        ResetStateVector();
    }

    template <typename Fn> bitCapInt BitCapIntAsStateVector(Fn operation)
    {
        SetStateVector();
        bitCapInt toRet = operation(stateVecUnit);
        ResetStateVector();

        return toRet;
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QBdtPtr dest);

    void ApplyControlledSingle(const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target);

    static size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)((perm >> bit) & 1U); }

    static bitCapInt RemovePower(bitCapInt perm, bitCapInt power)
    {
        bitCapInt mask = power - ONE_BCI;
        return (perm & mask) | ((perm >> ONE_BCI) & ~mask);
    }

public:
    QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QBdt(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QBdt({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem,
              deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold, separation_thresh)
    {
    }

    virtual bool isBinaryDecisionTree() { return true; };

    virtual void Finish()
    {
        if (stateVecUnit) {
            stateVecUnit->Finish();
        }
    };

    virtual bool isFinished() { return !stateVecUnit || stateVecUnit->isFinished(); }

    virtual void Dump()
    {
        if (stateVecUnit) {
            stateVecUnit->Dump();
        }
    }

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank.
    }

    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1)
    {
        root->Normalize(bdtQubitCount);
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QBdt>(toCompare));
    }
    virtual real1_f SumSqrDiff(QBdtPtr toCompare);

    virtual void SetPermutation(bitCapInt initState, complex phaseFac = CMPLX_DEFAULT_ARG);

    virtual QInterfacePtr Clone();

    virtual void GetQuantumState(complex* state);
    virtual void GetQuantumState(QInterfacePtr eng);
    virtual void SetQuantumState(const complex* state);
    virtual void SetQuantumState(QInterfacePtr eng);
    virtual void GetProbs(real1* outputProbs);

    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->SetAmplitude(perm, amp); });
    }

    using QInterface::Compose;
    virtual bitLenInt Compose(QBdtPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QBdt>(toCopy), start);
    }
    virtual bitLenInt Attach(QEnginePtr toCopy, bitLenInt start)
    {
        if (start == qubitCount) {
            return Attach(toCopy);
        }

        const bitLenInt origSize = qubitCount;
        ROL(origSize - start, 0, qubitCount);
        bitLenInt result = Attach(toCopy, qubitCount);
        ROR(origSize - start, 0, qubitCount);

        return result;
    }
    virtual bitLenInt Attach(QEnginePtr toCopy);
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        DecomposeDispose(start, dest->GetQubitCount(), std::dynamic_pointer_cast<QBdt>(dest));
    }
    virtual void Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, NULL); }

    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        DecomposeDispose(start, length, NULL);
    }

    virtual real1_f Prob(bitLenInt qubitIndex);
    virtual real1_f ProbAll(bitCapInt fullRegister);

    virtual std::map<bitCapInt, int> MultiShotMeasureMask(
        const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots)
    {
        Finish();
        QInterfacePtr unit = stateVecUnit ? stateVecUnit : MakeTempStateVector();
        return unit->MultiShotMeasureMask(qPowers, qPowerCount, shots);
    }

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);
    virtual bitCapInt MAll();

    virtual void Mtrx(const complex* mtrx, bitLenInt target);
    virtual void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target)
    {
        const complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
        if (!IS_NORM_0(ONE_CMPLX - topLeft)) {
            ApplyControlledSingle(mtrx, controls, controlLen, target);
            return;
        }

        std::unique_ptr<bitLenInt[]> lControls = std::unique_ptr<bitLenInt[]>(new bitLenInt[controlLen]);
        std::copy(controls, controls + controlLen, lControls.get());
        std::sort(lControls.get(), lControls.get() + controlLen);

        if (target < lControls[controlLen - 1U]) {
            std::swap(target, lControls[controlLen - 1U]);
        }

        ApplyControlledSingle(mtrx, lControls.get(), controlLen, target);
    }
    virtual void MCInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target)
    {
        const complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
        if (!IS_NORM_0(ONE_CMPLX - topRight) || !IS_NORM_0(ONE_CMPLX - bottomLeft)) {
            ApplyControlledSingle(mtrx, controls, controlLen, target);
            return;
        }

        std::vector<bitLenInt> controlVec(controlLen);
        std::copy(controls, controls + controlLen, controlVec.begin());
        std::sort(controlVec.begin(), controlVec.end());

        if (controlVec.back() < target) {
            ApplyControlledSingle(mtrx, controls, controlLen, target);
            return;
        }

        H(target);
        MCPhase(controls, controlLen, ONE_CMPLX, -ONE_CMPLX, target);
        H(target);
    }

    virtual bool ForceMParity(bitCapInt mask, bool result, bool doForce = true);
    virtual real1_f ProbParity(bitCapInt mask)
    {
        Finish();
        QInterfacePtr unit = stateVecUnit ? stateVecUnit : MakeTempStateVector();
        return unit->ProbParity(mask);
    }
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->FSim(theta, phi, qubitIndex1, qubitIndex2); });
    }
    virtual void CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->CSqrtSwap(controls, controlLen, qubit1, qubit2); });
    }
    virtual void AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2); });
    }
    virtual void CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->CISqrtSwap(controls, controlLen, qubit1, qubit2); });
    }
    virtual void AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2); });
    }

    virtual void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->CUniformParityRZ(controls, controlLen, mask, angle); });
    }

    virtual void PhaseParity(real1_f radians, bitCapInt mask)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->PhaseParity(radians, mask); });
    }

#if ENABLE_ALU
    virtual void INCDECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { eng->INCDECSC(toAdd, start, length, overflowIndex, carryIndex); });
    }
    virtual void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCDECSC(toAdd, start, length, carryIndex); });
    }
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCBCD(toAdd, start, length); });
    }
    virtual void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCDECBCDC(toAdd, start, length, carryIndex); });
    }
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->MUL(toMul, inOutStart, carryStart, length); });
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->DIV(toDiv, inOutStart, carryStart, length); });
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->MULModNOut(toMul, modN, inStart, outStart, length); });
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->IMULModNOut(toMul, modN, inStart, outStart, length); });
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->POWModNOut(base, modN, inStart, outStart, length); });
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { eng->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen); });
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { eng->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen); });
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { eng->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); });
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            eng->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { eng->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen); });
    }
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values, bool resetValue = true)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return eng->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
        });
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return eng->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return eng->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    virtual void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->Hash(start, length, values); });
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->PhaseFlipIfLess(greaterPerm, start, length); });
    }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex); });
    }
#endif
};
} // namespace Qrack
