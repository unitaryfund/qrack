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

#include "qbinary_decision_tree_node.hpp"
#include "qinterface.hpp"

#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
#include "common/dispatchqueue.hpp"
#endif
#include "mpsshard.hpp"

namespace Qrack {

class QBinaryDecisionTree;
typedef std::shared_ptr<QBinaryDecisionTree> QBinaryDecisionTreePtr;

class QBinaryDecisionTree : virtual public QInterface {
protected:
    std::vector<QInterfaceEngine> engines;
    int devID;
    QBinaryDecisionTreeNodePtr root;
    QInterfacePtr stateVecUnit;
#if ENABLE_QUNIT_CPU_PARALLEL
    DispatchQueue dispatchQueue;
#endif
    bitLenInt bdtThreshold;
    bitLenInt dispatchThreshold;
    bitCapIntOcl maxQPowerOcl;
    bool isFusionFlush;
    std::vector<MpsShardPtr> shards;

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        maxQPowerOcl = (bitCapIntOcl)maxQPower;
    }

    typedef std::function<void(void)> DispatchFn;
    virtual void Dispatch(bitCapInt workItemCount, DispatchFn fn)
    {
#if ENABLE_QUNIT_CPU_PARALLEL
        if ((workItemCount >= (bitCapIntOcl)(ONE_BCI << dispatchThreshold)) &&
            (workItemCount < GetParallelThreshold())) {
            dispatchQueue.dispatch(fn);
        } else {
            Finish();
            fn();
        }
#else
        fn();
#endif
    }

    QInterfacePtr MakeStateVector();

    QInterfacePtr MakeTempStateVector()
    {
        QInterfacePtr copyPtr = MakeStateVector();
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
        stateVecUnit = MakeStateVector();
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
    }

    template <typename Fn> bitCapInt BitCapIntAsStateVector(Fn operation)
    {
        SetStateVector();
        bitCapInt toRet = operation(stateVecUnit);

        return toRet;
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest);

    void Apply2x2OnLeaf(const complex* mtrx, QBinaryDecisionTreeNodePtr leaf, bitLenInt depth,
        bitCapInt highControlMask, bool isAnti, bool isParallel);

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

    void FlushBuffer(bitLenInt i)
    {
        MpsShardPtr shard = shards[i];
        if (!shard) {
            return;
        }

        shards[i] = NULL;
        isFusionFlush = true;
        Mtrx(shard->gate, i);
        isFusionFlush = false;
    }

    void FlushBuffers()
    {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            FlushBuffer(i);
        }
    }

    void DumpBuffers()
    {
        for (bitLenInt i = 0U; i < qubitCount; i++) {
            shards[i] = NULL;
        }
    }

public:
    QBinaryDecisionTree(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QBinaryDecisionTree(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QBinaryDecisionTree({ QINTERFACE_QPAGER }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold,
              separation_thresh)
    {
    }

    virtual bool isBinaryDecisionTree() { return true; };

    virtual void Finish()
    {
#if ENABLE_QUNIT_CPU_PARALLEL
        dispatchQueue.finish();
#endif
        if (stateVecUnit) {
            stateVecUnit->Finish();
        }
    };

    virtual bool isFinished()
    {
#if ENABLE_QUNIT_CPU_PARALLEL
        return dispatchQueue.isFinished() && (!stateVecUnit || stateVecUnit->isFinished());
#else
        return !stateVecUnit || stateVecUnit->isFinished();
#endif
    }

    virtual void Dump()
    {
#if ENABLE_QUNIT_CPU_PARALLEL
        dispatchQueue.dump();
#endif
    }

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank.
    }

    virtual void NormalizeState(real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        root->Normalize(qubitCount);
    }

    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QBinaryDecisionTree>(toCompare));
    }
    virtual real1_f SumSqrDiff(QBinaryDecisionTreePtr toCompare);

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
    virtual bitLenInt Compose(QBinaryDecisionTreePtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QBinaryDecisionTree>(toCopy), start);
    }

    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        DecomposeDispose(start, dest->GetQubitCount(), std::dynamic_pointer_cast<QBinaryDecisionTree>(dest));
    }
    virtual void Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, NULL); }

    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        if (stateVecUnit && ((qubitCount - length) <= bdtThreshold)) {
            stateVecUnit->Dispose(start, length, disposedPerm);
            shards.erase(shards.begin() + start, shards.begin() + start + length);
            SetQubitCount(qubitCount - length);

            return;
        }

        DecomposeDispose(start, length, NULL);
    }

    virtual real1_f Prob(bitLenInt qubitIndex);
    virtual real1_f ProbAll(bitCapInt fullRegister);

    virtual std::map<bitCapInt, int> MultiShotMeasureMask(
        const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots)
    {
        FlushBuffers();
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
