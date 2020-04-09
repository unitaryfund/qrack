//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <set>

#include "bitbuffer.hpp"
#include "qinterface.hpp"

// MODLEN returns "arg" modulo the capacity of "len" bits. We use it to check if an arithmetic "arg" is equivalent to
// the identity operator. (That is, we check if [addend] % [maxInt] == 0, such that we are effectively adding or
// subtracting 0.) If a method call is equivalent to the identity operator, it makes no difference, so we do not flush
// for it or buffer it.
#define MODLEN(arg, len) (arg & (pow2(len) - ONE_BCI))

namespace Qrack {

class QFusion;
typedef std::shared_ptr<QFusion> QFusionPtr;

class QFusion : public QInterface {
protected:
    static const bitLenInt MIN_FUSION_BITS = 3U;
    QInterfacePtr qReg;
    complex phaseFactor;
    bool doNormalize;

    std::vector<BitBufferPtr> bitBuffers;
    std::vector<std::set<bitLenInt>> bitControls;

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = pow2(qb);
        bitBuffers.resize(qb);
        bitControls.resize(qb);
    }

public:
    QFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceID = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> devList = {});
    QFusion(QInterfacePtr target);

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp);
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual void SetBit(bitLenInt qubitIndex, bool value);
    using QInterface::Compose;
    virtual bitLenInt Compose(QFusionPtr toCopy, bool isConsumed = false);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bool isConsumed = false)
    {
        return Compose(std::dynamic_pointer_cast<QFusion>(toCopy), isConsumed);
    }
    virtual bitLenInt Compose(QFusionPtr toCopy, bitLenInt start, bool isConsumed = false);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start, bool isConsumed = false)
    {
        return Compose(std::dynamic_pointer_cast<QFusion>(toCopy), start, isConsumed);
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        Decompose(start, length, std::dynamic_pointer_cast<QFusion>(dest));
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QFusionPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual bool TryDecompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        return TryDecompose(start, length, std::dynamic_pointer_cast<QFusion>(dest));
    }
    virtual bool TryDecompose(bitLenInt start, bitLenInt length, QFusionPtr dest);
    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubitIndex);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask);
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
    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true);
    virtual bitCapInt ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values);
    virtual bitCapInt ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true);

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);
    virtual void CDEC(
        bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
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
    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void CFullAdd(bitLenInt* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2,
        bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void CIFullAdd(bitLenInt* controls, bitLenInt controlLen, bitLenInt input1, bitLenInt input2,
        bitLenInt carryInSumOut, bitLenInt carryOut);

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
    virtual void PhaseFlip();

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    virtual real1 Prob(bitLenInt qubitIndex);
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation);
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation);
    virtual real1 ProbAll(bitCapInt fullRegister);

    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QFusion>(toCompare));
    }
    virtual bool ApproxCompare(QFusionPtr toCompare);

    virtual QInterfacePtr ReleaseEngine()
    {
        FlushAll();
        QInterfacePtr toRet = qReg;
        qReg = NULL;
        SetQubitCount(0);
        return toRet;
    }

    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual void Finish()
    {
        FlushAll();
        qReg->Finish();
    }

    virtual bool isFinished() { return (qReg == NULL) || qReg->isFinished(); }

    virtual QInterfacePtr Clone()
    {
        FlushAll();

        QInterfacePtr payload = qReg->Clone();

        return std::make_shared<QFusion>(payload);
    }

    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1);

protected:
    /** Buffer flush methods, to apply accumulated buffers when bits are checked for output or become involved in
     * nonbufferable operations */

    void FlushBit(const bitLenInt& qubitIndex);

    void FlushReg(const bitLenInt& start, const bitLenInt& length)
    {
        for (bitLenInt i = 0U; i < length; i++) {
            FlushBit(start + i);
        }
    }

    void FlushArray(const bitLenInt* bitList, const bitLenInt& bitListLen)
    {
        for (bitLenInt i = 0; i < bitListLen; i++) {
            FlushBit(bitList[i]);
        }
    }

    void FlushSet(std::set<bitLenInt> bitList)
    {
        std::set<bitLenInt>::iterator it;
        for (it = bitList.begin(); it != bitList.end(); it++) {
            FlushBit(*it);
        }
    }

    void FlushMask(const bitCapInt mask)
    {
        bitCapInt v = mask; // count the number of bits set in v
        bitCapInt oldV;
        bitCapInt power;
        bitLenInt length; // c accumulates the total bits set in v
        for (length = 0; v; length++) {
            oldV = v;
            v &= v - 1; // clear the least significant bit set
            power = (v ^ oldV) & oldV;
            if (power) {
                FlushBit(log2(power));
            }
        }
    }

    void FlushAll() { FlushReg(0, qubitCount); }

    /** Buffer discard methods, for when the state of a bit becomes irrelevant before a buffer flush */

    void DiscardBit(const bitLenInt& qubitIndex);

    inline void DiscardReg(const bitLenInt& start, const bitLenInt& length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            DiscardBit(start + i);
        }
    }

    inline void DiscardAll() { DiscardReg(0, qubitCount); }

    /** Method to compose arithmetic gates */
    void BufferArithmetic(bitLenInt* controls, bitLenInt controlLen, int toAdd, bitLenInt inOutStart, bitLenInt length);

    void EraseControls(std::vector<bitLenInt> controls, bitLenInt qubitIndex);
};
} // namespace Qrack
