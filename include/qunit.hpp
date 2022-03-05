//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qengineshard.hpp"

#if ENABLE_ALU
#include "qalu.hpp"
#endif

namespace Qrack {

class QUnit;
typedef std::shared_ptr<QUnit> QUnitPtr;

#if ENABLE_ALU
class QUnit : public QAlu, public QInterface {
#else
class QUnit : public QInterface {
#endif
protected:
    std::vector<QInterfaceEngine> engines;
    int devID;
    QEngineShardMap shards;
    complex phaseFactor;
    bool doNormalize;
    bool useHostRam;
    bool useRDRAND;
    bool isSparse;
    bool freezeBasis2Qb;
    bool isReactiveSeparate;
    bitLenInt thresholdQubits;
    real1_f separabilityThreshold;
    std::vector<int> deviceIDs;

    QInterfacePtr MakeEngine(bitLenInt length, bitCapInt perm);

public:
    QUnit(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devIDs = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QUnit(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devIDs = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QUnit({ QINTERFACE_STABILIZER_HYBRID }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devIDs, qubitThreshold,
              separation_thresh)
    {
    }

    virtual ~QUnit() { Dump(); }

    virtual void SetConcurrency(uint32_t threadsPerEngine)
    {
        QInterface::SetConcurrency(threadsPerEngine);
        ParallelUnitApply(
            [](QInterfacePtr unit, real1_f unused1, real1_f unused2, real1_f unused3, int32_t threadsPerEngine) {
                unit->SetConcurrency(threadsPerEngine);
                return true;
            },
            ZERO_R1, ZERO_R1, ZERO_R1, threadsPerEngine);
    }

    virtual void SetReactiveSeparate(bool isAggSep) { isReactiveSeparate = isAggSep; }
    virtual bool GetReactiveSeparate() { return isReactiveSeparate; }

    virtual void SetDevice(int dID, bool forceReInit = false);
    virtual int64_t GetDevice() { return devID; }

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
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    using QInterface::H;
    virtual void H(bitLenInt target);
    using QInterface::S;
    virtual void S(bitLenInt target);
    using QInterface::IS;
    virtual void IS(bitLenInt target);

    virtual void ZMask(bitCapInt mask) { PhaseParity(PI_R1, mask); }
    virtual void PhaseParity(real1 radians, bitCapInt mask);

    virtual void Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex);
    virtual void Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex);
    virtual void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    virtual void MCInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);
    virtual void MACPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    virtual void MACInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);
    virtual void Mtrx(const complex* mtrx, bitLenInt qubit);
    virtual void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    using QInterface::UniformlyControlledSingleBit;
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
        const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask);
    virtual void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle);
    virtual void CSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::ForceM;
    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, bool doApply = true);
    using QInterface::ForceMReg;
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true, bool doApply = true);
    virtual bitCapInt MAll();
    virtual std::map<bitCapInt, int> MultiShotMeasureMask(
        const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots);
    virtual void MultiShotMeasureMask(
        const bitCapInt* qPowers, bitLenInt qPowerCount, unsigned shots, unsigned* shotsArray);

    /** @} */

#if ENABLE_ALU
    virtual bool M(bitLenInt q) { return QInterface::M(q); }
    virtual void X(bitLenInt q) { QInterface::X(q); }

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) { QInterface::DEC(toSub, start, length); }
    virtual void DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::DECS(toSub, start, length, overflowIndex);
    }
    virtual void CDEC(
        bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        QInterface::CDEC(toSub, inOutStart, length, controls, controlLen);
    }
    virtual void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCDECC(toAdd, start, length, carryIndex);
    }

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCDECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCDECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, const unsigned char* values);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);

    /** @} */
#endif

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual real1_f Prob(bitLenInt qubit);
    virtual real1_f ProbAll(bitCapInt fullRegister);
    virtual real1_f ProbParity(bitCapInt mask);
    virtual bool ForceMParity(bitCapInt mask, bool result, bool doForce = true);
    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QUnit>(toCompare));
    }
    virtual real1_f ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset = 0);

    virtual real1_f SumSqrDiff(QUnitPtr toCompare);
    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1);
    virtual void Finish();
    virtual bool isFinished();
    virtual void Dump()
    {
        for (size_t i = 0; i < shards.size(); i++) {
            shards[i].unit = NULL;
        }
    }
    virtual bool isClifford(bitLenInt qubit) { return shards[qubit].isClifford(); };

    virtual bool TrySeparate(const bitLenInt* qubits, bitLenInt length, real1_f error_tol);
    virtual bool TrySeparate(bitLenInt qubit);
    virtual bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2);

    virtual QInterfacePtr Clone();

    /** @} */

protected:
    virtual complex GetAmplitudeOrProb(bitCapInt perm, bool isProb);

    virtual void XBase(bitLenInt target);
    virtual void YBase(bitLenInt target);
    virtual void ZBase(bitLenInt target);
    virtual real1_f ProbBase(bitLenInt qubit);

    virtual bool TrySeparateClifford(bitLenInt qubit);

#if ENABLE_ALU
    typedef void (QAlu::*INCxFn)(bitCapInt, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QAlu::*INCxxFn)(bitCapInt, bitLenInt, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QAlu::*CMULFn)(bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    typedef void (QAlu::*CMULModFn)(bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, const bitLenInt* controls, bitLenInt controlLen);
    void INT(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex, bool hasCarry,
        std::vector<bitLenInt> controlVec = std::vector<bitLenInt>());
    void INTS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex,
        bool hasCarry);
    void INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void INCxx(
        INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index);
    QInterfacePtr CMULEntangle(std::vector<bitLenInt> controlVec, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, std::vector<bitLenInt>* controlsMapped);
    std::vector<bitLenInt> CMULEntangle(
        std::vector<bitLenInt> controlVec, bitLenInt start, bitCapInt carryStart, bitLenInt length);
    void CMULx(CMULFn fn, bitCapInt toMod, bitLenInt start, bitLenInt carryStart, bitLenInt length,
        std::vector<bitLenInt> controlVec);
    void xMULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length, bool inverse);
    void CxMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen, bool inverse);
    void CMULModx(CMULModFn fn, bitCapInt toMod, bitCapInt modN, bitLenInt start, bitLenInt carryStart,
        bitLenInt length, std::vector<bitLenInt> controlVec);
    bool INTCOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex);
    bool INTSOptimize(bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt overflowIndex);
    bool INTSCOptimize(
        bitCapInt toMod, bitLenInt start, bitLenInt length, bool isAdd, bitLenInt carryIndex, bitLenInt overflowIndex);
    bitCapInt GetIndexedEigenstate(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values);
    bitCapInt GetIndexedEigenstate(bitLenInt start, bitLenInt length, const unsigned char* values);
#endif

    virtual QInterfacePtr Entangle(std::vector<bitLenInt> bits);
    virtual QInterfacePtr Entangle(std::vector<bitLenInt*> bits);
    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length, bool isForProb = false);
    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2);
    virtual QInterfacePtr EntangleRange(
        bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3);
    virtual QInterfacePtr EntangleAll(bool isForProb = false)
    {
        QInterfacePtr toRet = EntangleRange(0, qubitCount, isForProb);
        OrderContiguous(toRet);
        return toRet;
    }

    virtual QInterfacePtr CloneBody(QUnitPtr copyPtr);

    virtual bool CheckBitsPermutation(bitLenInt start, bitLenInt length = 1);
    virtual bitCapInt GetCachedPermutation(bitLenInt start, bitLenInt length);
    virtual bitCapInt GetCachedPermutation(const bitLenInt* bitArray, bitLenInt length);
    virtual bool CheckBitsPlus(bitLenInt qubitIndex, bitLenInt length);

    virtual QInterfacePtr EntangleInCurrentBasis(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    typedef bool (*ParallelUnitFn)(QInterfacePtr unit, real1_f param1, real1_f param2, real1_f param3, int32_t param4);
    bool ParallelUnitApply(ParallelUnitFn fn, real1_f param1 = ZERO_R1, real1_f param2 = ZERO_R1,
        real1_f param3 = ZERO_R1, int32_t param4 = 0);

    virtual bool SeparateBit(bool value, bitLenInt qubit);

    void OrderContiguous(QInterfacePtr unit);

    virtual void Detach(bitLenInt start, bitLenInt length, QUnitPtr dest);

    struct QSortEntry {
        bitLenInt bit;
        bitLenInt mapped;
        bool operator<(const QSortEntry& rhs) { return mapped < rhs.mapped; }
        bool operator>(const QSortEntry& rhs) { return mapped > rhs.mapped; }
    };
    void SortUnit(QInterfacePtr unit, std::vector<QSortEntry>& bits, bitLenInt low, bitLenInt high);

    bool TrimControls(const bitLenInt* controls, bitLenInt controlLen, std::vector<bitLenInt>& output, bool anti);

    template <typename CF>
    void ApplyEitherControlled(
        std::vector<bitLenInt> controlVec, const std::vector<bitLenInt> targets, CF cfn, bool isPhase);

    void ClampShard(bitLenInt qubit)
    {
        QEngineShard& shard = shards[qubit];
        if (!shard.ClampAmps() || !shard.unit) {
            return;
        }

        if (IS_NORM_0(shard.amp1)) {
            SeparateBit(false, qubit);
        } else if (IS_NORM_0(shard.amp0)) {
            SeparateBit(true, qubit);
        }
    }

    void ShardAI(bitLenInt qubit, real1_f azimuth, real1_f inclination)
    {
        real1 cosineA = (real1)cos(azimuth);
        real1 sineA = (real1)sin(azimuth);
        real1 cosineI = (real1)cos(inclination / 2);
        real1 sineI = (real1)sin(inclination / 2);
        complex expA = complex(cosineA, sineA);
        complex expNegA = complex(cosineA, -sineA);
        complex mtrx[4] = { cosineI, -expNegA * sineI, expA * sineI, cosineI };

        QEngineShard& shard = shards[qubit];

        const complex Y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        ClampShard(qubit);
    }

    void TransformX2x2(const complex* mtrxIn, complex* mtrxOut);
    void TransformXInvert(complex topRight, complex bottomLeft, complex* mtrxOut);
    void TransformY2x2(const complex* mtrxIn, complex* mtrxOut);
    void TransformYInvert(complex topRight, complex bottomLeft, complex* mtrxOut);
    void TransformPhase(complex topLeft, complex bottomRight, complex* mtrxOut);

    void RevertBasisX(bitLenInt i)
    {
        QEngineShard& shard = shards[i];
        if (shard.pauliBasis != PauliX) {
            // Recursive call that should be blocked,
            // or already in target basis.
            return;
        }

        ConvertZToX(i);
    }

    void RevertBasisY(bitLenInt i)
    {
        QEngineShard& shard = shards[i];

        if (shard.pauliBasis != PauliY) {
            // Recursive call that should be blocked,
            // or already in target basis.
            return;
        }

        shard.pauliBasis = PauliX;

        const complex mtrx[4] = { ((real1)(ONE_R1 / 2)) * (ONE_CMPLX + I_CMPLX),
            ((real1)(ONE_R1 / 2)) * (ONE_CMPLX - I_CMPLX), ((real1)(ONE_R1 / 2)) * (ONE_CMPLX - I_CMPLX),
            ((real1)(ONE_R1 / 2)) * (ONE_CMPLX + I_CMPLX) };

        if (shard.unit) {
            shard.unit->Mtrx(mtrx, shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        complex Y0 = shard.amp0;

        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        ClampShard(i);
    }

    void RevertBasis1Qb(bitLenInt i)
    {
        QEngineShard& shard = shards[i];

        if (shard.pauliBasis == PauliY) {
            ConvertYToZ(i);
        } else {
            RevertBasisX(i);
        }
    }

    void RevertBasisToX1Qb(bitLenInt i)
    {
        QEngineShard& shard = shards[i];
        if (shard.pauliBasis == PauliZ) {
            ConvertZToX(i);
        } else if (shard.pauliBasis == PauliY) {
            RevertBasisY(i);
        }
    }

    void RevertBasisToY1Qb(bitLenInt i)
    {
        QEngineShard& shard = shards[i];
        if (shard.pauliBasis == PauliZ) {
            ConvertZToY(i);
        } else if (shard.pauliBasis == PauliX) {
            ConvertXToY(i);
        }
    }

    virtual void ConvertZToX(bitLenInt i)
    {
        QEngineShard& shard = shards[i];

        // WARNING: Might be called when shard is in either Z or X basis
        shard.pauliBasis = (shard.pauliBasis == PauliX) ? PauliZ : PauliX;

        if (shard.unit) {
            shard.unit->H(shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        complex tempAmp1 = SQRT1_2_R1 * (shard.amp0 - shard.amp1);
        shard.amp0 = SQRT1_2_R1 * (shard.amp0 + shard.amp1);
        shard.amp1 = tempAmp1;
        ClampShard(i);
    }
    virtual void ConvertXToY(bitLenInt i)
    {
        QEngineShard& shard = shards[i];

        shard.pauliBasis = PauliY;

        const complex mtrx[4] = { ((real1)(ONE_R1 / 2)) * (ONE_CMPLX - I_CMPLX),
            ((real1)(ONE_R1 / 2)) * (ONE_CMPLX + I_CMPLX), ((real1)(ONE_R1 / 2)) * (ONE_CMPLX + I_CMPLX),
            ((real1)(ONE_R1 / 2)) * (ONE_CMPLX - I_CMPLX) };

        if (shard.unit) {
            shard.unit->Mtrx(mtrx, shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        complex Y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        ClampShard(i);
    }
    virtual void ConvertYToZ(bitLenInt i)
    {
        QEngineShard& shard = shards[i];

        shard.pauliBasis = PauliZ;

        const complex mtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(SQRT1_2_R1, ZERO_R1),
            complex(ZERO_R1, SQRT1_2_R1), complex(ZERO_R1, -SQRT1_2_R1) };

        if (shard.unit) {
            shard.unit->Mtrx(mtrx, shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        complex Y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        ClampShard(i);
    }
    virtual void ConvertZToY(bitLenInt i)
    {
        QEngineShard& shard = shards[i];

        shard.pauliBasis = PauliY;

        const complex mtrx[4] = { complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, -SQRT1_2_R1),
            complex(SQRT1_2_R1, ZERO_R1), complex(ZERO_R1, SQRT1_2_R1) };

        if (shard.unit) {
            shard.unit->Mtrx(mtrx, shard.mapped);
        }

        if (shard.isPhaseDirty || shard.isProbDirty) {
            shard.isProbDirty = true;
            return;
        }

        complex Y0 = shard.amp0;
        shard.amp0 = (mtrx[0] * Y0) + (mtrx[1] * shard.amp1);
        shard.amp1 = (mtrx[2] * Y0) + (mtrx[3] * shard.amp1);
        ClampShard(i);
    }

    enum RevertExclusivity { INVERT_AND_PHASE = 0, ONLY_INVERT = 1, ONLY_PHASE = 2 };
    enum RevertControl { CONTROLS_AND_TARGETS = 0, ONLY_CONTROLS = 1, ONLY_TARGETS = 2 };
    enum RevertAnti { CTRL_AND_ANTI = 0, ONLY_CTRL = 1, ONLY_ANTI = 2 };

    void ApplyBuffer(PhaseShardPtr phaseShard, bitLenInt control, bitLenInt target, bool isAnti);
    void ApplyBufferMap(bitLenInt bitIndex, ShardToPhaseMap bufferMap, RevertExclusivity exclusivity, bool isControl,
        bool isAnti, const std::set<bitLenInt>& exceptPartners, bool dumpSkipped);
    void RevertBasis2Qb(bitLenInt i, RevertExclusivity exclusivity = INVERT_AND_PHASE,
        RevertControl controlExclusivity = CONTROLS_AND_TARGETS, RevertAnti antiExclusivity = CTRL_AND_ANTI,
        const std::set<bitLenInt>& exceptControlling = {}, const std::set<bitLenInt>& exceptTargetedBy = {},
        bool dumpSkipped = false, bool skipOptimized = false);

    void Flush0Eigenstate(bitLenInt i)
    {
        shards[i].DumpControlOf();
        if (randGlobalPhase) {
            shards[i].DumpSamePhaseAntiControlOf();
        }
        RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_ANTI);
    }
    void Flush1Eigenstate(bitLenInt i)
    {
        shards[i].DumpAntiControlOf();
        if (randGlobalPhase) {
            shards[i].DumpSamePhaseControlOf();
        }
        RevertBasis2Qb(i, INVERT_AND_PHASE, ONLY_CONTROLS, ONLY_CTRL);
    }
    void ToPermBasis(bitLenInt i)
    {
        RevertBasis1Qb(i);
        RevertBasis2Qb(i);
    }
    void ToPermBasis(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            RevertBasis1Qb(start + i);
        }
        for (bitLenInt i = 0; i < length; i++) {
            RevertBasis2Qb(start + i);
        }
    }
    void ToPermBasisAll() { ToPermBasis(0, qubitCount); }
    void ToPermBasisProb(bitLenInt qubit)
    {
        RevertBasis1Qb(qubit);
        RevertBasis2Qb(qubit, ONLY_INVERT, ONLY_TARGETS);
    }
    void ToPermBasisProb(bitLenInt start, bitLenInt length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            RevertBasis1Qb(start + i);
        }
        for (bitLenInt i = 0; i < length; i++) {
            RevertBasis2Qb(start + i, ONLY_INVERT, ONLY_TARGETS);
        }
    }
    void ToPermBasisProb() { ToPermBasisProb(0, qubitCount); }
    void ToPermBasisMeasure(bitLenInt qubit)
    {
        RevertBasis1Qb(qubit);
        RevertBasis2Qb(qubit, ONLY_INVERT);
        RevertBasis2Qb(qubit, ONLY_PHASE, ONLY_CONTROLS);

        shards[qubit].DumpMultiBit();
    }
    void ToPermBasisMeasure(bitLenInt start, bitLenInt length)
    {
        if ((start == 0) && (length == qubitCount)) {
            ToPermBasisAllMeasure();
            return;
        }

        std::set<bitLenInt> exceptBits;
        for (bitLenInt i = 0; i < length; i++) {
            exceptBits.insert(start + i);
        }
        for (bitLenInt i = 0; i < length; i++) {
            RevertBasis1Qb(start + i);
        }
        for (bitLenInt i = 0; i < length; i++) {
            RevertBasis2Qb(start + i, ONLY_INVERT);
            RevertBasis2Qb(start + i, ONLY_PHASE, ONLY_CONTROLS, CTRL_AND_ANTI, exceptBits);
            shards[start + i].DumpMultiBit();
        }
    }
    void ToPermBasisAllMeasure()
    {
        for (bitLenInt i = 0; i < qubitCount; i++) {
            RevertBasis1Qb(i);
        }
        for (bitLenInt i = 0; i < qubitCount; i++) {
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
        for (bitLenInt i = 0; i < (bitLenInt)bitIndices.size(); i++) {
            shards[bitIndices[i]].MakeDirty();
        }
    }

    void EndEmulation(bitLenInt target)
    {
        QEngineShard& shard = shards[target];
        if (shard.unit) {
            return;
        }

        if (norm(shard.amp1) <= FP_NORM_EPSILON) {
            shard.unit = MakeEngine(1, 0);
        } else if (norm(shard.amp0) <= FP_NORM_EPSILON) {
            shard.unit = MakeEngine(1, 1);
        } else {
            complex bitState[2] = { shard.amp0, shard.amp1 };
            shard.unit = MakeEngine(1, 0);
            shard.unit->SetQuantumState(bitState);
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

    void CommuteH(bitLenInt bitIndex);

    void OptimizePairBuffers(bitLenInt control, bitLenInt target, bool anti);
};

} // namespace Qrack
