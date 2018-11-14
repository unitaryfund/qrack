//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"

namespace Qrack {

class QFusion;
typedef std::shared_ptr<QFusion> QFusionPtr;

class QFusion : public QInterface {
protected:
    static const bitLenInt MIN_FUSION_BITS = 3U;
    QInterfacePtr qReg;
    QInterfaceEngine engineType;
    std::shared_ptr<std::default_random_engine> rand_generator;

    std::vector<std::shared_ptr<complex[4]>> bitBuffers;

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = 1 << qubitCount;
        bitBuffers.resize(qb);
    }

    virtual void NormalizeState(real1 nrm = -999.0)
    {
        // Intentionally left blank
    }

public:
    QFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        std::shared_ptr<std::default_random_engine> rgp = nullptr);

    virtual void SetQuantumState(complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetPermutation(bitCapInt perm);
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    using QInterface::Cohere;
    virtual bitLenInt Cohere(QInterfacePtr toCopy) { return Cohere(std::dynamic_pointer_cast<QFusion>(toCopy)); }
    virtual bitLenInt Cohere(QFusionPtr toCopy);
    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        Decohere(start, length, std::dynamic_pointer_cast<QFusion>(dest));
    }
    virtual void Decohere(bitLenInt start, bitLenInt length, QFusionPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
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
    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, real1 nrmlzr = ONE_R1);
    virtual bitCapInt ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true);

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
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);

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
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    virtual void CopyState(QInterfacePtr orig) { return CopyState(std::dynamic_pointer_cast<QFusion>(orig)); }
    virtual void CopyState(QFusionPtr orig);
    virtual bool IsPhaseSeparable(bool forceCheck = false);
    virtual real1 Prob(bitLenInt qubitIndex);
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation);
    virtual real1 ProbAll(bitCapInt fullRegister);

protected:
    inline void FlushBit(bitLenInt qubitIndex)
    {
        if (bitBuffers[qubitIndex]) {
            qReg->ApplySingleBit(bitBuffers[qubitIndex].get(), true, qubitIndex);
            bitBuffers[qubitIndex] = NULL;
        }
    }
    inline void FlushReg(const bitLenInt& start, const bitLenInt& length)
    {
        for (bitLenInt i = 0U; i < length; i++) {
            FlushBit(start + i);
        }
    }
    inline void FlushAll() { FlushReg(0, qubitCount); }
    inline void DiscardReg(const bitLenInt& start, const bitLenInt& length)
    {
        for (bitLenInt i = 0; i < length; i++)
            bitBuffers[start + i] = NULL;
    }
    inline void DiscardAll() { DiscardReg(0, qubitCount); }
    inline void FlushList(const bitLenInt* bitList, const bitLenInt& bitListLen)
    {
        for (bitLenInt i = 0; i < bitListLen; i++)
            FlushBit(bitList[i]);
    }
};
} // namespace Qrack
