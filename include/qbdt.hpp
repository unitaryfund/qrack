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
    bitCapIntOcl maxQPowerOcl;
    bitCapIntOcl treeLevelPowerOcl;
    bitLenInt treeLevelCount;
    bitLenInt attachedQubitCount;
    bitLenInt bdtQubitCount;
    std::vector<MpsShardPtr> shards;

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        maxQPowerOcl = (bitCapIntOcl)maxQPower;
        bdtQubitCount = qubitCount - attachedQubitCount;
        treeLevelCount = attachedQubitCount ? (bdtQubitCount + 1U) : bdtQubitCount;
        treeLevelPowerOcl = pow2Ocl(treeLevelCount);
    }

    typedef std::function<void(void)> DispatchFn;
    virtual void Dispatch(bitCapInt workItemCount, DispatchFn fn) { fn(); }

    QInterfacePtr MakeStateVector(bitLenInt qbCount, bitCapInt perm = 0U);
    QBdtQInterfaceNodePtr MakeQEngineNode(complex scale, bitLenInt qbCount, bitCapInt perm = 0U)
    {
        return std::make_shared<QBdtQInterfaceNode>(
            scale, std::dynamic_pointer_cast<QEngine>(MakeStateVector(qbCount, perm)));
    }

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

        FlushBuffers();
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

    void Apply2x2OnLeaf(const complex* mtrx, QBdtNodeInterfacePtr leaf, bitLenInt depth, bitCapInt highControlMask,
        bool isAnti, bool isParallel);

    template <typename Fn> void ApplySingle(const complex* mtrx, bitLenInt target, Fn leafFunc);
    template <typename Lfn>
    void ApplyControlledSingle(const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target,
        bool isAnti, Lfn leafFunc);

    static size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)((perm >> bit) & 1U); }

    static bitCapInt RemovePower(bitCapInt perm, bitCapInt power)
    {
        bitCapInt mask = power - ONE_BCI;
        return (perm & mask) | ((perm >> ONE_BCI) & ~mask);
    }

    void FlushBuffer(bitLenInt i);

    void FlushBuffers()
    {
        for (bitLenInt i = 0; i < bdtQubitCount; i++) {
            FlushBuffer(i);
        }
        Finish();
    }

    void DumpBuffers()
    {
        for (bitLenInt i = 0U; i < bdtQubitCount; i++) {
            shards[i] = NULL;
        }
    }

    bool CheckControlled(
        const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target, bool isAnti);

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

    virtual void Dump() {}

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank.
    }

    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1)
    {
        root->Normalize(treeLevelCount);
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
    virtual bitLenInt Attach(QEnginePtr toCopy, bitLenInt start);
    virtual bitLenInt Attach(QEnginePtr toCopy) { return Attach(toCopy, qubitCount); }
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
        FlushBuffers();
        Finish();
        QInterfacePtr unit = stateVecUnit ? stateVecUnit : MakeTempStateVector();
        return unit->MultiShotMeasureMask(qPowers, qPowerCount, shots);
    }

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);
    virtual bitCapInt MAll();

    virtual void Mtrx(const complex* mtrx, bitLenInt target);
    virtual void Phase(complex topLeft, complex bottomRight, bitLenInt target);
    virtual void Invert(complex topRight, complex bottomLeft, bitLenInt target);
    virtual void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);

    virtual bool ForceMParity(bitCapInt mask, bool result, bool doForce = true);
    virtual real1_f ProbParity(bitCapInt mask)
    {
        FlushBuffers();
        Finish();
        QInterfacePtr unit = stateVecUnit ? stateVecUnit : MakeTempStateVector();
        return unit->ProbParity(mask);
    }

    virtual void Swap(bitLenInt low, bitLenInt high)
    {
        if (high < low) {
            std::swap(low, high);
        }

        CNOT(low, high);

        if ((low < bdtQubitCount) && (bdtQubitCount <= high)) {
            // Low qubits are QBdt; high qubits are QEngine.
            // Target qubit must be in QEngine, if acting with QEngine.
            H(high);
            H(low);
            CNOT(low, high);
            H(high);
            H(low);
        } else {
            CNOT(high, low);
        }

        CNOT(low, high);
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
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INC(toAdd, start, length); });
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->CINC(toAdd, inOutStart, length, controls, controlLen); });
    }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCC(toAdd, start, length, carryIndex); });
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCS(toAdd, start, length, overflowIndex); });
    }
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCSC(toAdd, start, length, overflowIndex, carryIndex); });
    }
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCSC(toAdd, start, length, carryIndex); });
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->DECC(toSub, start, length, carryIndex); });
    }
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->DECSC(toSub, start, length, overflowIndex, carryIndex); });
    }
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->DECSC(toSub, start, length, carryIndex); });
    }
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCBCD(toAdd, start, length); });
    }
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->INCBCDC(toAdd, start, length, carryIndex); });
    }
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->DECBCDC(toSub, start, length, carryIndex); });
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
