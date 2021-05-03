//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <cfloat>
#include <random>
#include <unordered_set>

#include "qengineshard.hpp"

namespace Qrack {

class QUnit;
typedef std::shared_ptr<QUnit> QUnitPtr;

class QUnit : public QInterface {
protected:
    QInterfaceEngine engine;
    QInterfaceEngine subEngine;
    int devID;
    QEngineShardMap shards;
    complex phaseFactor;
    bool doNormalize;
    bool useHostRam;
    bool useRDRAND;
    bool isSparse;
    bool freezeBasisH;
    bool freezeBasis2Qb;
    bool freezeTrySeparate;
    bool isReactiveSeparate;
    bool isPagingSuppressed;
    bool canSuppressPaging;
    bitLenInt thresholdQubits;
    bitLenInt pagingThresholdQubits;
    real1_f separabilityThreshold;
    std::vector<int> deviceIDs;

    QInterfacePtr MakeEngine(bitLenInt length, bitCapInt perm);

    virtual void TurnOnPaging();
    virtual void TurnOffPaging();
    virtual void ConvertPaging(const bool& isPaging)
    {
        if (!canSuppressPaging) {
            return;
        }

        if (isPaging) {
            TurnOnPaging();
        } else {
            TurnOffPaging();
        }
    }

public:
    QUnit(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QUnit(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, ignored, qubitThreshold, separation_thresh)
    {
    }

    QUnit(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QUnit(QINTERFACE_OPTIMAL_G0_CHILD, QINTERFACE_OPTIMAL_G1_CHILD, qBitCount, initState, rgp, phaseFac, doNorm,
              randomGlobalPhase, useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, ignored,
              qubitThreshold, separation_thresh)
    {
    }

    virtual ~QUnit() { Dump(); }

    virtual void SetConcurrency(uint32_t threadsPerEngine)
    {
        ParallelUnitApply(
            [](QInterfacePtr unit, real1_f unused1, real1_f unused2, int32_t threadsPerEngine) {
                unit->SetConcurrency(threadsPerEngine);
                return true;
            },
            ZERO_R1, ZERO_R1, threadsPerEngine);
    }

    virtual void SetReactiveSeparate(const bool& isAggSep) { isReactiveSeparate = isAggSep; }
    virtual bool GetReactiveSeparate() { return isReactiveSeparate; }

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp);
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);
    using QInterface::Compose;
    virtual bitLenInt Compose(QUnitPtr toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QUnit>(toCopy)); }
    virtual bitLenInt Compose(QUnitPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QUnit>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QUnit>(dest));
    }
    virtual void Decompose(bitLenInt start, QUnitPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    using QInterface::H;
    virtual void H(bitLenInt target);
    using QInterface::X;
    virtual void X(bitLenInt target);
    using QInterface::Z;
    virtual void Z(bitLenInt target);
    using QInterface::CNOT;
    virtual void CNOT(bitLenInt control, bitLenInt target);
    using QInterface::AntiCNOT;
    virtual void AntiCNOT(bitLenInt control, bitLenInt target);
    using QInterface::CCNOT;
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    using QInterface::AntiCCNOT;
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    using QInterface::CZ;
    virtual void CZ(bitLenInt control, bitLenInt target);
    using QInterface::CCZ;
    virtual void CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target);
    using QInterface::CH;
    virtual void CH(bitLenInt control, bitLenInt target);

    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex);
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex);
    virtual void ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);
    virtual void ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);
    virtual void ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);
    virtual void ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);
    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubit);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    using QInterface::UniformlyControlledSingleBit;
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask);
    virtual void CUniformParityRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle);
    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    using QInterface::ForceM;
    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, bool doApply = true);
    using QInterface::ForceMReg;
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true, bool doApply = true);
    virtual bitCapInt MAll();

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values);
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2);
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual real1_f Prob(bitLenInt qubit);
    virtual real1_f ProbAll(bitCapInt fullRegister);
    virtual real1_f ProbParity(const bitCapInt& mask);
    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true);
    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QUnit>(toCompare));
    }
    virtual real1_f SumSqrDiff(QUnitPtr toCompare);
    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    virtual void NormalizeState(real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_threshold = REAL1_DEFAULT_ARG);
    virtual void Finish();
    virtual bool isFinished();
    virtual void Dump();
    virtual bool isClifford(const bitLenInt& qubit) { return shards[qubit].isClifford(); };

    virtual bool TrySeparate(bitLenInt* qubits, bitLenInt length, real1_f error_tol);
    virtual bool TrySeparate(bitLenInt qubit);
    virtual bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2);

    virtual QInterfacePtr Clone();

    /** @} */

protected:
    virtual void XBase(const bitLenInt& target);
    virtual void YBase(const bitLenInt& target);
    virtual void ZBase(const bitLenInt& target);
    virtual real1_f ProbBase(const bitLenInt& qubit);

    typedef void (QInterface::*INCxFn)(bitCapInt, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QInterface::*INCxxFn)(bitCapInt, bitLenInt, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QInterface::*CMULFn)(bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    typedef void (QInterface::*CMULModFn)(bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    void INT(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex, bool hasCarry,
        bitLenInt* controls = NULL, bitLenInt controlLen = 0);
    void INTS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex,
        bool hasCarry);
    void INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void INCxx(
        INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index);
    QInterfacePtr CMULEntangle(std::vector<bitLenInt> controlVec, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, std::vector<bitLenInt>* controlsMapped);
    std::vector<bitLenInt> CMULEntangle(
        std::vector<bitLenInt> controlVec, bitLenInt start, bitCapInt carryStart, bitLenInt length);
    void CMULx(CMULFn fn, bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length, bitLenInt* controls,
        bitLenInt controlLen);
    void xMULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length, bool inverse);
    void CxMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen, bool inverse);
    void CMULModx(CMULModFn fn, bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, std::vector<bitLenInt> controlVec);
    bool CArithmeticOptimize(bitLenInt* controls, bitLenInt controlLen, std::vector<bitLenInt>* controlVec);
    bool INTCOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex);
    bool INTSOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt overflowIndex);
    bool INTSCOptimize(
        bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex, bitLenInt overflowIndex);

    virtual QInterfacePtr Entangle(std::vector<bitLenInt> bits);
    virtual QInterfacePtr Entangle(std::vector<bitLenInt*> bits);
    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length);
    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2);
    virtual QInterfacePtr EntangleRange(
        bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3);
    virtual QInterfacePtr EntangleAll()
    {
        QInterfacePtr toRet = EntangleRange(0, qubitCount);
        OrderContiguous(toRet);
        return toRet;
    }

    virtual QInterfacePtr CloneBody(QUnitPtr copyPtr);

    virtual bool CheckBitPermutation(const bitLenInt& qubitIndex, const bool& inCurrentBasis = false);
    virtual bool CheckBitsPermutation(
        const bitLenInt& start, const bitLenInt& length, const bool& inCurrentBasis = false);
    virtual bitCapInt GetCachedPermutation(const bitLenInt& start, const bitLenInt& length);
    virtual bitCapInt GetCachedPermutation(const bitLenInt* bitArray, const bitLenInt& length);
    virtual bool CheckBitsPlus(const bitLenInt& qubitIndex, const bitLenInt& length);

    virtual QInterfacePtr EntangleInCurrentBasis(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    typedef bool (*ParallelUnitFn)(QInterfacePtr unit, real1_f param1, real1_f param2, int32_t param3);
    bool ParallelUnitApply(ParallelUnitFn fn, real1_f param1 = ZERO_R1, real1_f param2 = ZERO_R1, int32_t param3 = 0);

    virtual void SeparateBit(bool value, bitLenInt qubit, bool doDispose = true);

    void OrderContiguous(QInterfacePtr unit);

    virtual void Detach(bitLenInt start, bitLenInt length, QUnitPtr dest);

    struct QSortEntry {
        bitLenInt bit;
        bitLenInt mapped;
        bool operator<(const QSortEntry& rhs) { return mapped < rhs.mapped; }
        bool operator>(const QSortEntry& rhs) { return mapped > rhs.mapped; }
    };
    void SortUnit(QInterfacePtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high);

    template <typename CF, typename F>
    void ApplyEitherControlled(const bitLenInt* controls, const bitLenInt& controlLen,
        const std::vector<bitLenInt> targets, const bool& anti, CF cfn, F f, const bool& isPhase = false,
        const bool& isInvert = false, const bool& inCurrentBasis = false);

    bitCapInt GetIndexedEigenstate(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);
    bitCapInt GetIndexedEigenstate(bitLenInt start, bitLenInt length, unsigned char* values);

    void TransformX2x2(const complex* mtrxIn, complex* mtrxOut);
    void TransformXInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut);
    void TransformY2x2(const complex* mtrxIn, complex* mtrxOut);
    void TransformYInvert(const complex& topRight, const complex& bottomLeft, complex* mtrxOut);
    void TransformPhase(const complex& topLeft, const complex& bottomRight, complex* mtrxOut);

    void RevertBasisX(const bitLenInt& i)
    {
        if (freezeBasisH || !shards[i].isPauliX) {
            // Recursive call that should be blocked,
            // or already in target basis.
            return;
        }

        freezeBasisH = true;
        H(i);
        shards[i].isPauliX = false;
        freezeBasisH = false;
    }

    void RevertBasisY(const bitLenInt& i)
    {
        if (freezeBasisH || !shards[i].isPauliY) {
            // Recursive call that should be blocked,
            // or already in target basis.
            return;
        }

        shards[i].isPauliY = false;
        shards[i].isPauliX = true;
        freezeBasisH = true;
        S(i);
        freezeBasisH = false;
    }

    void RevertBasis1Qb(const bitLenInt& i)
    {
        RevertBasisY(i);
        RevertBasisX(i);
    }

    enum RevertExclusivity { INVERT_AND_PHASE = 0, ONLY_INVERT = 1, ONLY_PHASE = 2 };
    enum RevertControl { CONTROLS_AND_TARGETS = 0, ONLY_CONTROLS = 1, ONLY_TARGETS = 2 };
    enum RevertAnti { CTRL_AND_ANTI = 0, ONLY_CTRL = 1, ONLY_ANTI = 2 };

    void ApplyBuffer(PhaseShardPtr phaseShard, const bitLenInt& control, const bitLenInt& target, const bool& isAnti);
    void ApplyBufferMap(const bitLenInt& bitIndex, ShardToPhaseMap bufferMap, const RevertExclusivity& exclusivity,
        const bool& isControl, const bool& isAnti, std::set<bitLenInt> exceptPartners, const bool& dumpSkipped);
    void RevertBasis2Qb(const bitLenInt& i, const RevertExclusivity& exclusivity = INVERT_AND_PHASE,
        const RevertControl& controlExclusivity = CONTROLS_AND_TARGETS,
        const RevertAnti& antiExclusivity = CTRL_AND_ANTI, std::set<bitLenInt> exceptControlling = {},
        std::set<bitLenInt> exceptTargetedBy = {}, const bool& dumpSkipped = false, const bool& skipOptimized = false);

    void Flush0Eigenstate(const bitLenInt& i)
    {
        shards[i].DumpControlOf();
        if (randGlobalPhase) {
            shards[i].DumpSamePhaseAntiControlOf();
        }
        RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_ANTI);
    }
    void Flush1Eigenstate(const bitLenInt& i)
    {
        shards[i].DumpAntiControlOf();
        if (randGlobalPhase) {
            shards[i].DumpSamePhaseControlOf();
        }
        RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_CTRL);
    }
    void ToPermBasis(const bitLenInt& i)
    {
        RevertBasisY(i);
        RevertBasisX(i);
        RevertBasis2Qb(i);
    }
    void ToPermBasis(const bitLenInt& start, const bitLenInt& length)
    {
        bitLenInt i;
        for (i = 0; i < length; i++) {
            RevertBasisY(start + i);
        }
        for (i = 0; i < length; i++) {
            RevertBasisX(start + i);
        }
        for (i = 0; i < length; i++) {
            RevertBasis2Qb(start + i);
        }
    }
    void ToPermBasisAll() { ToPermBasis(0, qubitCount); }
    void ToPermBasisProb(const bitLenInt& qubit)
    {
        RevertBasis1Qb(qubit);
        RevertBasis2Qb(qubit, ONLY_INVERT, ONLY_TARGETS);
    }
    void ToPermBasisProb(const bitLenInt& start, const bitLenInt& length)
    {
        bitLenInt i;
        for (i = 0; i < length; i++) {
            RevertBasis1Qb(start + i);
        }
        for (i = 0; i < length; i++) {
            RevertBasis2Qb(start + i, ONLY_INVERT, ONLY_TARGETS);
        }
    }
    void ToPermBasisMeasure(const bitLenInt& qubit)
    {
        if (qubitCount == 1U) {
            ToPermBasisAllMeasure();
            return;
        }

        RevertBasis1Qb(qubit);
        RevertBasis2Qb(qubit, ONLY_INVERT);
        RevertBasis2Qb(qubit, ONLY_PHASE, ONLY_CONTROLS);

        shards[qubit].DumpMultiBit();
    }
    void ToPermBasisMeasure(const bitLenInt& start, const bitLenInt& length)
    {
        if ((start == 0) && (length == qubitCount)) {
            ToPermBasisAllMeasure();
            return;
        }

        bitLenInt i;

        std::set<bitLenInt> exceptBits;
        for (i = 0; i < length; i++) {
            exceptBits.insert(start + i);
        }
        for (i = 0; i < length; i++) {
            RevertBasis1Qb(start + i);
        }
        for (i = 0; i < length; i++) {
            RevertBasis2Qb(start + i, ONLY_INVERT);
            RevertBasis2Qb(start + i, ONLY_PHASE, ONLY_CONTROLS, CTRL_AND_ANTI, exceptBits);
            shards[start + i].DumpMultiBit();
        }
    }
    void ToPermBasisAllMeasure()
    {
        bitLenInt i;
        for (i = 0; i < qubitCount; i++) {
            RevertBasis1Qb(i);
        }
        for (i = 0; i < qubitCount; i++) {
            shards[i].ClearInvertPhase();
            RevertBasis2Qb(i, ONLY_INVERT);
            shards[i].DumpMultiBit();
        }
    }

    void DirtyShardRange(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            shards[start + i].MakeDirty();
        }
    }

    void DirtyShardRangePhase(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            shards[start + i].isPhaseDirty = true;
        }
    }

    void DirtyShardIndexVector(std::vector<bitLenInt> bitIndices)
    {
        for (bitLenInt i = 0; i < bitIndices.size(); i++) {
            shards[bitIndices[i]].MakeDirty();
        }
    }

    void EndEmulation(QEngineShard& shard)
    {
        if (!shard.unit) {
            complex bitState[2] = { shard.amp0, shard.amp1 };
            shard.unit = MakeEngine(1, 0);
            shard.unit->SetQuantumState(bitState);
        }
    }

    void EndEmulation(const bitLenInt& target)
    {
        QEngineShard& shard = shards[target];
        EndEmulation(shard);
    }

    void EndEmulation(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            EndEmulation(start + i);
        }
    }

    void EndEmulation(bitLenInt* bitIndices, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            EndEmulation(bitIndices[i]);
        }
    }

    void EndAllEmulation()
    {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            EndEmulation(i);
        }
    }

    bitLenInt FindShardIndex(QEngineShardPtr shard)
    {
        shard->found = true;
        for (bitLenInt i = 0; i < shards.size(); i++) {
            if (shards[i].found) {
                shard->found = false;
                return i;
            }
        }
        shard->found = false;
        return shards.size();
    }

    void CommuteH(const bitLenInt& bitIndex);

    void OptimizePairBuffers(const bitLenInt& control, const bitLenInt& target, const bool& anti);

    void CacheSingleQubitShard(bitLenInt target);
};

} // namespace Qrack
