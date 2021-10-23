//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QEngineShard is the atomic qubit unit of the QUnit mapper. "PhaseShard" optimizations are basically just a very
// specific "gate fusion" type optimization, where multiple gates are composed into single product gates before
// application to the state vector, to reduce the total number of gates that need to be applied. Rather than handling
// this as a "QFusion" layer optimization, which will typically sit BETWEEN a base QEngine set of "shards" and a QUnit
// that owns them, this particular gate fusion optimization can be avoid representational entanglement in QUnit in the
// first place, which QFusion would not help with. Alternatively, another QFusion would have to be in place ABOVE the
// QUnit layer, (with QEngine "below,") for this to work. Additionally, QFusion is designed to handle more general gate
// fusion, not specifically controlled phase gates, which are entirely commuting among each other and possibly a
// jumping-off point for further general "Fourier basis" optimizations which should probably reside in QUnit, analogous
// to the |+>/|-> basis changes QUnit takes advantage of for "H" gates.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qbinary_decision_tree_node.hpp"
#include "qengine_cpu.hpp"
#include "qinterface.hpp"

namespace Qrack {

class QBinaryDecisionTree;
typedef std::shared_ptr<QBinaryDecisionTree> QBinaryDecisionTreePtr;

class QBinaryDecisionTree : virtual public QInterface {
protected:
    QBinaryDecisionTreeNodePtr root;

    template <typename Fn> void GetTraversal(Fn getLambda);
    template <typename Fn> void SetTraversal(Fn setLambda);
    template <typename Fn> void ProductSetTraversal(Fn setLambda);
    template <typename Fn> void ExecuteAsQEngineCPU(Fn operation)
    {
        QInterfacePtr copyPtr = std::make_shared<QEngineCPU>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
            randGlobalPhase, false, -1, hardware_rand_generator != NULL, false, amplitudeFloor);

        GetQuantumState(copyPtr);
        operation(copyPtr);
        SetQuantumState(copyPtr);
    }

    template <typename Fn> bitCapInt ResultAsQEngineCPU(Fn operation)
    {
        QInterfacePtr copyPtr = std::make_shared<QEngineCPU>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
            randGlobalPhase, false, -1, hardware_rand_generator != NULL, false, amplitudeFloor);

        GetQuantumState(copyPtr);
        bitCapInt toRet = operation(copyPtr);
        SetQuantumState(copyPtr);

        return toRet;
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QBinaryDecisionTreePtr dest);

    void Apply2x2OnLeaves(const complex* mtrx, QBinaryDecisionTreeNodePtr* leaf0, QBinaryDecisionTreeNodePtr* leaf1);

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
        : QBinaryDecisionTree({}, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold, separation_thresh)
    {
    }

    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank.
    }

    virtual void NormalizeState(real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank.
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
    virtual void SetAmplitude(bitCapInt perm, complex amp);

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

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt target);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);

    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true);
    virtual real1_f ProbParity(const bitCapInt& mask);

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true)
    {
        return ResultAsQEngineCPU([&](QInterfacePtr eng) {
            return eng->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
        });
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return ResultAsQEngineCPU([&](QInterfacePtr eng) {
            return eng->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return ResultAsQEngineCPU([&](QInterfacePtr eng) {
            return eng->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->Hash(start, length, values); });
    }

    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->FSim(theta, phi, qubitIndex1, qubitIndex2); });
    }

    virtual void CUniformParityRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->CUniformParityRZ(controls, controlLen, mask, angle); });
    }

    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->PhaseFlipIfLess(greaterPerm, start, length); });
    }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex); });
    }

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->INC(toAdd, start, length); });
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->CINC(toAdd, inOutStart, length, controls, controlLen); });
    }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->INCC(toAdd, start, length, carryIndex); });
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->INCS(toAdd, start, length, overflowIndex); });
    }
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->INCSC(toAdd, start, length, overflowIndex, carryIndex); });
    }
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->INCSC(toAdd, start, length, carryIndex); });
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->DECC(toSub, start, length, carryIndex); });
    }
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->DECSC(toSub, start, length, overflowIndex, carryIndex); });
    }
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->DECSC(toSub, start, length, carryIndex); });
    }
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->INCBCD(toAdd, start, length); });
    }
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->INCBCDC(toAdd, start, length, carryIndex); });
    }
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->DECBCDC(toSub, start, length, carryIndex); });
    }
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->MUL(toMul, inOutStart, carryStart, length); });
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->DIV(toDiv, inOutStart, carryStart, length); });
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->MULModNOut(toMul, modN, inStart, outStart, length); });
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->IMULModNOut(toMul, modN, inStart, outStart, length); });
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) { eng->POWModNOut(base, modN, inStart, outStart, length); });
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngineCPU(
            [&](QInterfacePtr eng) { eng->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen); });
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngineCPU(
            [&](QInterfacePtr eng) { eng->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen); });
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngineCPU(
            [&](QInterfacePtr eng) { eng->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen); });
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngineCPU([&](QInterfacePtr eng) {
            eng->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsQEngineCPU(
            [&](QInterfacePtr eng) { eng->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen); });
    }
};
} // namespace Qrack
