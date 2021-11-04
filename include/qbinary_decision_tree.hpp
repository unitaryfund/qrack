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
#include "qengine.hpp"

#include "common/dispatchqueue.hpp"
#include "common/parallel_for.hpp"

namespace Qrack {

class QBinaryDecisionTree;
typedef std::shared_ptr<QBinaryDecisionTree> QBinaryDecisionTreePtr;

class QBinaryDecisionTree : virtual public QInterface, public ParallelFor {
protected:
    std::vector<QInterfaceEngine> engines;
    int devID;
    QBinaryDecisionTreeNodePtr root;
#if ENABLE_QUNIT_CPU_PARALLEL
    DispatchQueue dispatchQueue;
#endif
    bitLenInt pStridePow;

    typedef std::function<void(void)> DispatchFn;
    virtual void Dispatch(bitCapInt workItemCount, DispatchFn fn)
    {
#if ENABLE_QUNIT_CPU_PARALLEL
        if (workItemCount < GetParallelThreshold()) {
            dispatchQueue.dispatch(fn);
        } else {
            Finish();
            fn();
        }
#else
        fn();
#endif
    }

    QEnginePtr MakeEngine();

    template <typename Fn> void GetTraversal(Fn getLambda);
    template <typename Fn> void SetTraversal(Fn setLambda);
    template <typename Fn> void ExecuteAsQEngine(Fn operation)
    {
        Finish();

        QEnginePtr copyPtr = MakeEngine();

        GetQuantumState(copyPtr);
        operation(copyPtr);
        SetQuantumState(copyPtr);
    }

    template <typename Fn> bitCapInt BitCapIntAsQEngine(Fn operation)
    {
        Finish();

        QEnginePtr copyPtr = MakeEngine();

        GetQuantumState(copyPtr);
        bitCapInt toRet = operation(copyPtr);
        SetQuantumState(copyPtr);

        return toRet;
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest);

    void Apply2x2OnLeaf(const complex* mtrx, QBinaryDecisionTreeNodePtr leaf, bitLenInt depth, bool isParallel,
        bitCapInt highControlMask, bool isAnti);

    template <typename Fn> void ApplySingle(bitLenInt target, Fn leafFunc);
    template <typename Lfn>
    void ApplyControlledSingle(bool isAnti, std::shared_ptr<complex[]> mtrx, const bitLenInt* controls,
        const bitLenInt& controlLen, const bitLenInt& target, Lfn leafFunc);

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
        : QBinaryDecisionTree({ QINTERFACE_HYBRID }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
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
    };

    virtual bool isFinished()
    {
#if ENABLE_QUNIT_CPU_PARALLEL
        return dispatchQueue.isFinished();
#else
        return true;
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
    virtual void GetQuantumState(QEnginePtr eng);
    virtual void SetQuantumState(const complex* state);
    virtual void SetQuantumState(QInterfacePtr eng);
    virtual void GetProbs(real1* outputProbs);

    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->SetAmplitude(perm, amp); });
    }

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
        DecomposeDispose(start, length, NULL);
    }

    virtual real1_f Prob(bitLenInt qubitIndex);
    virtual real1_f ProbAll(bitCapInt fullRegister);

    virtual std::map<bitCapInt, int> MultiShotMeasureMask(
        const bitCapInt* qPowers, const bitLenInt qPowerCount, const unsigned int shots)
    {
        Finish();

        QEnginePtr copyPtr = MakeEngine();

        GetQuantumState(copyPtr);
        return copyPtr->MultiShotMeasureMask(qPowers, qPowerCount, shots);
    }

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true, bool doApply = true)
    {
        return BitCapIntAsQEngine(
            [&](QInterfacePtr eng) { return eng->ForceMReg(start, length, result, doForce, doApply); });
    }
    virtual bitCapInt MAll();

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt target);
    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt target);
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt target);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);

    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true);
    virtual real1_f ProbParity(const bitCapInt& mask)
    {
        Finish();

        QEnginePtr copyPtr = MakeEngine();

        GetQuantumState(copyPtr);
        return copyPtr->ProbParity(mask);
    }

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true)
    {
        return BitCapIntAsQEngine([&](QInterfacePtr eng) {
            return eng->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
        });
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return BitCapIntAsQEngine([&](QInterfacePtr eng) {
            return eng->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return BitCapIntAsQEngine([&](QInterfacePtr eng) {
            return eng->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->Hash(start, length, values); });
    }

    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->SqrtSwap(qubitIndex1, qubitIndex2); });
    }
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->ISqrtSwap(qubitIndex1, qubitIndex2); });
    }
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->ISwap(qubitIndex1, qubitIndex2); });
    }
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->FSim(theta, phi, qubitIndex1, qubitIndex2); });
    }
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->CSqrtSwap(controls, controlLen, qubit1, qubit2); });
    }
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->AntiCSqrtSwap(controls, controlLen, qubit1, qubit2); });
    }
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->CISqrtSwap(controls, controlLen, qubit1, qubit2); });
    }
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->AntiCISqrtSwap(controls, controlLen, qubit1, qubit2); });
    }

    virtual void CUniformParityRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->CUniformParityRZ(controls, controlLen, mask, angle); });
    }

    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->PhaseFlipIfLess(greaterPerm, start, length); });
    }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex); });
    }

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->INC(toAdd, start, length); });
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->CINC(toAdd, inOutStart, length, controls, controlLen); });
    }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->INCC(toAdd, start, length, carryIndex); });
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->INCS(toAdd, start, length, overflowIndex); });
    }
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->INCSC(toAdd, start, length, overflowIndex, carryIndex); });
    }
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->INCSC(toAdd, start, length, carryIndex); });
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->DECC(toSub, start, length, carryIndex); });
    }
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->DECSC(toSub, start, length, overflowIndex, carryIndex); });
    }
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->DECSC(toSub, start, length, carryIndex); });
    }
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->INCBCD(toAdd, start, length); });
    }
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->INCBCDC(toAdd, start, length, carryIndex); });
    }
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->DECBCDC(toSub, start, length, carryIndex); });
    }
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->MUL(toMul, inOutStart, carryStart, length); });
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->DIV(toDiv, inOutStart, carryStart, length); });
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->MULModNOut(toMul, modN, inStart, outStart, length); });
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->IMULModNOut(toMul, modN, inStart, outStart, length); });
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->POWModNOut(base, modN, inStart, outStart, length); });
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngine(
            [&](QInterfacePtr eng) { eng->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen); });
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngine(
            [&](QInterfacePtr eng) { eng->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen); });
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngine(
            [&](QInterfacePtr eng) { eng->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); });
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) {
            eng->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngine(
            [&](QInterfacePtr eng) { eng->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen); });
    }
    virtual void PhaseParity(real1_f radians, bitCapInt mask)
    {
        ExecuteAsQEngine([&](QInterfacePtr eng) { eng->PhaseParity(radians, mask); });
    }
};
} // namespace Qrack
